# Clipzyme wrapper for EMULaToR (enzyme retrieval)

This wrapper trains Clipzyme with its published settings on the EMULaToR enzyme retrieval splits.
It runs several seeds on every split folder. It does no tuning.
Run every command from the submodule root (`baselines/Clipzyme`).

## Atom maps

Clipzyme needs atom-mapped reactions. The wrapper reads the EMULaToR LocalMapper maps from
`data/processed/datasets/enzyme_retrieval_dataset/reaction_atom_mappings.parquet` (`--atom_map_file`).
It joins on `rxn_smiles` and keeps rows with `atom_mapping_status == "mapped"`.
Rows without a map, and rows that the original featurization cannot read, are counted per split in `inputs/stats.json`.
`inputs/` is reused only when the atom-map file, the three source parquet files (path, size, mtime), the row limits and `INPUTS_VERSION` match `stats.json["fingerprint"]`. Otherwise it is rebuilt.
The original featurization needs a map number on every atom, so reactions with unmapped atoms are dropped.
Train and val rows that fail are dropped. Test rows that fail stay in the test set and are scored as misses (empty ranking at every k).
The wrapper never runs a mapper.

## Environment

Use the `clipzyme` env from `environment.yml` (`PY` below). It needs torch, pytorch-lightning, torch-geometric,
torch-scatter, fair-esm 2.0.0 (not the `esm` package of ESM3), rdkit, pandas, pyarrow, tensorboard, gitpython,
comet_ml, p_tqdm, wget and rxn-chem-utils. The build commands are in `commands.txt`.
ESM-2 weights come from `<out_root>/weights/torch_hub` (`--hub_dir`). That folder holds the torch hub checkout
`facebookresearch_esm_v2.0.0`, `trusted_list`, and `checkpoints/esm2_t33_650M_UR50D.pt` (2.6 GB) with its
`-contact-regression.pt` file. The wrapper never downloads them.

## Data and splits

Splits come from `data/processed/datasets/enzyme_retrieval_dataset`. The wrapper finds every folder that has
`train/val/test.parquet` with the columns `rxn_smiles`, `ec_number` and `sequence`. It prints the folders it skips.
A leading `with_protein_information/` is removed from the name. `<group>/threshold_*` gives group and name.
A folder without `threshold_*` gets the name `all`. Nothing is hard coded, so new folders are used when they appear.

## Layout

`OUT` is `--out_root` (default `outputs/baselines/Clipzyme`).

```
OUT/enzyme_retrieval/<group>/<name>/inputs/     Clipzyme JSON, pickles, train candidates, val and test queries, stats.json
OUT/enzyme_retrieval/<group>/<name>/seed_<s>/   bestmodel.pth, lastmodel.pth, checkpoint_last.pt, predictions.parquet,
                                                metrics.json, run_summary.json, train_summary.json, tensorboard/
OUT/enzyme_retrieval/summary_by_seed.csv        rebuilt from predictions.parquet by emulator_bench.results
OUT/enzyme_retrieval/summary.csv                mean and std (n-1) over seeds
OUT/weights/torch_hub/                          ESM-2 weights
OUT/logs/enzyme_retrieval/                      one log file per prepare and per run
```

`checkpoint_last.pt` holds the full training state (optimizer, scheduler). It is about 8 GB.
`bestmodel.pth` and `lastmodel.pth` hold weights only (about 2.6 GB each).
A run resumes from `checkpoint_last.pt` when it is started again.

## Training

`train.py` starts the original `scripts/main.py` through `run_main.py` with the settings of `default_hparams.json`.
`run_main.py` only renames the Lightning checkpoint files, resumes from `checkpoint_last.pt`, and counts the
samples that the original dataset code keeps. It changes no model code. With `--checkpoint_esm` it wraps the ESM-2 layer call in activation recomputation.
One run uses one GPU. The optimizer step uses batch 8 and 8 accumulation steps (global batch 64).
`--checkpoint_esm` recomputes the ESM-2 layers in the backward pass. The numbers stay the same. Batch 8 needs it on a 23 GB GPU.
`--fast` applies the exact speed-ups of `fast_esm.py` at run time (the upstream package stays unchanged):
ESM-2 attention through `scaled_dot_product_attention` and `F.gelu`; inside each micro-batch the ESM forward runs in length-sorted
chunks of about 1024 tokens and the outputs return in the original order; ESM-2 layers recompute activations in the backward pass
only when a micro-batch has more than 4000 tokens (with `--checkpoint_esm` they always recompute); vectorised `get_reverse_edge_indices`.
Micro-batches, batch order and contrastive negatives do not change. In `evaluate`, `--fast` also uses SDPA and encodes the candidate
enzymes in order of length. Results equal the default path up to floating-point noise (fp32 tests in `tests/test_fast_esm.py`).
`--amp` is off by default (original precision, bf16). With `--amp`: bf16 on Ampere or newer, fp16 on older GPUs, fp32 on CPU.

## Frozen ESM-2 (`--frozen_esm`)

This path follows `configs/train/clip_esm_frozen.json`: `clip_freeze_esm`, ESM-2 650M never trains. The reaction DMPNNs,
`ln_final` and `logit_scale` train. The frozen values are in the `frozen_hparams` section of `default_hparams.json`:
10 epochs, weight decay 0 (the config sets none, so the code default), and 32 samples per optimizer step. `--epochs` replaces the epoch count.
`from_checkpoint` and `cache_path` are not used. The scheduler quirk of the default path stays: warm-up lasts 100 epochs.

The model reads one vector per protein: the mean of ESM layer 33 over the residues (without CLS and EOS), then `ln_final`.
ESM-2 has no dropout, so the vector is the same at every step. `esm_cache.py` computes it once per unique sequence
(at most 650 aa, train, val and test of every split) in fp32 without TF32, in length-sorted chunks of about 24k tokens, and stores it
under `OUT/embeddings/esm_frozen_v1/` (`manifest.json` with the recipe, `part_*.pt` files keyed by the sha256 of the sequence).
A part is written atomically, so a stopped build continues. A different recipe is refused. Run it before training:

```
CUDA_VISIBLE_DEVICES=0 $PY -m emulator_bench.esm_cache --shard 0 --shards 3   # one process per GPU
```

With `--frozen_esm`, `train.py` and `evaluate.py` check the cache, load it onto the GPU and return `cache[sequence]` instead of
running ESM-2. The ESM-2 weights are loaded and dropped at model creation, so seed, initial weights and batch draws equal the
original frozen run, and checkpoints hold no ESM-2 weights (evaluate with `--frozen_esm` too).

Mapping of 4 GPUs to one GPU. In the original run each GPU takes 8 samples and contrasts them against all 32 gathered features of the step.
The gathered features carry no gradient, so they are constants, and DDP averages the 4 gradients. The wrapper loads 32 samples per step.
`virtual_ranks.py` encodes the reactions of each virtual rank of 8 separately (one padded batch of 32 would need a dense
`[graphs, atoms, atoms, 1280]` tensor sized by the largest graph of the whole step), builds the 32 protein vectors, and takes the mean of the
4 per-rank losses, each against the 32 constant features, in one backward pass. The model has no batch-dependent layer (no batch norm, dropout 0),
so encoding in ranks equals one batch up to floating-point noise. Validation uses the same loss on batches of 32.
`tests/test_frozen_esm.py` compares the gradient with four real gloo processes that run the original `CLIPLoss`, and the rank-wise encoder with one batch.

## Evaluation

`evaluate.py` loads `bestmodel.pth` and embeds every unique enzyme of the candidate pool and every test reaction.
The pool holds the proteins of train, val and test (at most 650 aa).
Each test reaction ranks the ECs of the pool. The score of an EC is the best cosine to any pool enzyme with that EC.
It saves the top 50 ECs per reaction in `predictions.parquet`.
Level-L accuracy at k is the share of true ECs of a reaction that the top-k ECs match up to level L (L = 4 is the exact EC).
Level 4 to level 1 map to the `X.X.X.X` to `X.-.-.-` columns of the paper. Values are given for k = 1, 3, 5, 10, 20, 30, 40, 50.
`results.py` rebuilds the summary CSVs from the predictions and checks them against `metrics.json`.

## Commands

See `commands.txt`.
