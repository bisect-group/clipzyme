# Clipzyme wrapper report

## 1. Method summary

Clipzyme is a contrastive model. An ESM-2 650M encoder embeds the enzyme. A 5-layer chemprop-style graph network embeds the atom-mapped reaction (reactant graph and product graph).
It trains with a CLIP loss on (reaction, enzyme) pairs. Task here: enzyme retrieval. A test reaction ranks ECs, and the score is EC-level accuracy (levels 4 to 1).

## 2. Original defaults

| Setting | Value | Source |
|---|---|---|
| batch size | 8 per GPU, 4 GPUs, accumulate 2 (64) | configs/train/clip_esm.json; paper App. D (64) |
| precision | bf16 | clip_esm.json; paper App. D |
| epochs | 20 | clip_esm.json (paper says about 30; code wins) |
| lr, weight decay | not in clip_esm.json (code defaults 0.001 and 0); paper 1e-4 and 0.05 | paper App. D; other configs (clip_egnn.json) |
| optimizer | AdamW, betas 0.9 0.999 | clip_esm.json; clipzyme/learning/optimizers/basic.py |
| scheduler | linear_warmup_cosine_lr, warm-up 100, start 1e-6, min 1e-5, init 1e-4 | clip_esm.json |
| monitor | val_clip_quantile | clip_esm.json |
| max protein length | 650 | clip_esm.json |
| remove_duplicate_reactions | true | clip_esm.json |
| ESM-2 | esm2_t33_650M_UR50D, trained (not frozen) | clip_esm.json |

All values are in `default_hparams.json` with a `sources` entry for each.

## 3. What the wrapper does

- `dataset_adapter.py` finds the retrieval splits on disk, reads the LocalMapper atom maps, runs the original featurization to check each row, drops rows that fail (counted per split in `inputs/stats.json`), and writes the EnzymeMap-style JSON and pickles that the original dataset class reads.
- `train.py` and `run_main.py` call the original `scripts/main.py` with the hparams. Only checkpoint file names, resume and size counts are added by hooks.
- `evaluate.py` loads `bestmodel.pth` with the original `CLIPZyme` class and uses its feature extractors. It ranks the ECs of the candidate pool (train, val and test proteins) per test reaction.
- `results.py` writes `summary_by_seed.csv` and `summary.csv` (mean and std over seeds, n-1).
- `run_all.py` is the launcher (`--gpus`, `--runs_per_gpu`, `--amp`, `--seeds`, `--split_groups`, `--dry_run`).

## 4. Files

Added: `atom_maps.py`, `default_hparams.json`, `run_all.py`, `run_main.py`, `commands.txt`, `REPORT.md`, `tests/test_emulator_bench.py`.
Rewritten: `utils.py`, `dataset_adapter.py`, `train.py`, `evaluate.py`, `results.py`, `README.md`, `queue_multiple_seeds.sh`.
Removed: `queue_pipeline.py`, `cache_features.py`.
Kept as found: `clipzyme/lightning/base.py` (drops `optimizer_idx` for Lightning 2) and `clipzyme/utils/loading.py` (`type(model).load_from_checkpoint`). Both look correct.

## 5. Bugs fixed in the old wrapper

- It ran RXNMapper by itself. Now it only reads the EMULaToR LocalMapper maps.
- It cut val and test sequences to 650 aa. Now such rows are not truncated.
- It removed duplicates by (reaction, sequence, EC) and turned the original dedup off. Now the original dedup runs.
- It ranked EC centers over proteins of train, val and test. Now each EC scores by its best protein; the pool is every split protein (user decision).
- Wrong layout (`runs/`, `results/`), seed default 0 in evaluation, hard-coded cache paths, log scraping, own precision flag. All replaced by the spec layout and flags.

## 6. Deviations from the original

- One GPU per run, accumulate 8 instead of 2 on 4 GPUs. Global batch is 64 but the in-batch negatives are 8 per step, not 32. This is forced by one GPU per run.
- lr 1e-4 and weight decay 0.05 (paper, not in clip_esm.json).
- Candidate pool is every protein of the split: train, val and test (at most 650 aa), as in the original screening set.
- Test queries are not length filtered. The original filters more than 650 aa for every split.
- EC score is the best cosine to any pool enzyme with that EC. It is not an EC center score.
- Reactions are stored in the processed form (sorted, `[H+]` removed). The original uses a raw string on its screening path.
- Reaction JSON is built by the wrapper from the LocalMapper maps.
- k = 1 is the main value. The paper does not state k.
- Rows without a usable atom map. The original featurizer needs a map number on every atom, so rows with partial maps cannot be featurized (user decision: no patch). Train and val rows that fail are dropped and counted per split (`no_atom_map_rows`, `failed_featurization_rows` in `inputs/stats.json`). Test rows that fail stay in the test set and are scored as misses: the reaction gets an empty ranking, so it counts as a miss for every k and level. `scored_as_miss_rows` in `stats.json`, `scored_as_miss_queries` in `metrics.json` and `scored_as_miss` in the summary CSVs give the count. This scores Clipzyme on the same test set as the other retrieval models. Test rows with no map at all are handled the same way.
- Original quirk kept: `scheduler_interval` is "epoch". The 100 warm-up "steps" are 100 epochs, so the lr stays near 2e-5 in 20 epochs and the cosine phase is never reached.
- Original dedup key is `reaction|uniprot` over the whole JSON. It ignores EC.
- Val queries are not evaluated. The best epoch uses the native `val_clip_quantile`.

## 7. Commands

See `commands.txt`. All splits and seeds:

```
PY=/path/to/clipzyme/bin/python
$PY -m emulator_bench.run_all --gpus 0 1 2            # splits on disk x seeds 666 777 888
$PY -m emulator_bench.run_all --gpus 0 1 2 --runs_per_gpu 2 --amp
```

Splits found now: `enzyme_sequence_splits/threshold_0.4`, `random_splits_grouped_rxn_key/all`, `uniprot_time_splits/all`. The DRFP split has no `sequence` column and is skipped.

## 8. Smoke result

Done on CPU with 64/16/16 real rows (old atom-map cache), seed 666, 2 epochs, A10 GPUs not used.
After the map and featurization check: train 61, val 15, test 15 rows (3, 1 and 1 failed featurization).
Train, evaluate and `results.py` ran end to end. Test queries: 8 reactions. Accuracy at k = 1: level 4 0.0, level 1 12.5. At k = 10: level 4 12.5, level 1 100. These numbers are random-level and only prove the code path.
Not run: real data, GPU, `--amp`, resume from `checkpoint_last.pt`.

## 9. Equivalence

Run A: original `scripts/main.py` with the `clip_esm.json` flags and the same overrides as the wrapper (paths, one GPU, accumulate 8, lr, wd, seed 666, 2 epochs). Run B: the wrapper. Same data (A reads the wrapper dataset.json).

| Check | Result |
|---|---|
| resolved args | equal except paths, logger names, and unused fields (split_type, split_probs, save_hiddens; `assign_splits` is false) |
| train loss, epoch 0 and 1 | 2.0896757, 2.0427366 in both |
| val_clip_quantile | 0.4345239, 0.4511905 in both |
| val loss | 2.0129605, 2.0123210 in both |
| best epoch | 1 in both |
| best and last weights | 586 tensors, 0 differ, max abs diff 0.0 |

Verdict: EQUIVALENT-EXCEPT-DOCUMENTED. Differences: the JSON conversion of the reactions is done by the wrapper (the original needs an EnzymeMap JSON), and the evaluation is new code (the original has no EC retrieval metric). Feature equality and batch order follow from equal data and seed, and the equal losses confirm them.

## 11. Speed-ups

Measured on one A10 (GPU 3, 23 GB), bf16 (original precision), real rows from `random_splits_grouped_rxn_key` (2.2k train samples), seed 666, batch 8, accumulate 8.

- ESM-2 is fine-tuned (`freeze_esm` false, `clip_esm.json:36`; `fair_esm.py:29,54`). Per-sequence ESM caching is not allowed.
- Batch 8 does not fit a 23 GB A10 (CUDA out of memory in the first forward). Batch 4 without recomputation also fails. The new flag `--checkpoint_esm` (hook in `run_main.py`) recomputes each ESM-2 layer in the backward pass. With it batch 8 runs, with 20.2 GB used on the GPU.
- Exactness of `--checkpoint_esm`: tiny run (batch 2, accumulate 4, 2 epochs, seed 666), 586 weight tensors. Two repeats without the flag differ by up to 2.7e-5 (bf16 kernels are not bit-reproducible). The run with the flag differs from the first repeat by up to 3.1e-5. Train loss (0.694), val loss and val_clip_quantile (0.227) are equal in all three runs. The change is exact within kernel noise.
- Cost: 1.65 s per micro-batch of 8 with the flag (slope of 24, 60 and 80 batch runs; 1.65 to 1.69 s per batch in the 240 batch run). That is 4.8 samples/s, 13.2 s per optimizer step of 64.
- Loader: 53 ms per sample and 11 ms per collate in one process, almost all in `from_mapped_smiles` (complete-edge lists in Python). One worker or 12 workers give the same run time (148 s and 152 s for 60 batches), so the run is GPU bound. Caching reaction graphs gives no gain. It was not built.
- Full run cost: one epoch of 1.1M rows is about 63 h, so 20 epochs are about 53 days per run on one A10. The real row count after dropping rows without a usable map is lower. Scale with 4.8 samples/s. `runs_per_gpu` is 1 (20 GB per run).
- 80 GB GPU: not measured. Batch 8 should fit without `--checkpoint_esm`, which would remove the recompute cost (about one third of the step). Not tested.
- No exact speed-up found beyond this. Non-exact options not measured: attention through SDPA, smaller micro-batch (changes the in-batch negatives).
- The equivalence of the wrapper against the original (section 9) is unchanged: the hooks do not change model code. The earlier CPU result stands.

## 12. Open issues

- Atom maps. The wrapper reads the LocalMapper file `reaction_atom_mappings.parquet` (rows with status "mapped", join on `rxn_smiles`). The original featurization (`from_mapped_smiles`, `get_bond_changes`) needs a map number on every atom. About 33 to 36 percent of the maps in the partial file (23000 rows) have unmapped atoms, so those rows fail featurization. The user decided to keep the original featurizer: train and val rows are dropped, test rows are scored as misses (see Deviations).
- Weights: `outputs/baselines/Clipzyme/weights/torch_hub` is a copy of the read-only old folder. It holds `checkpoints/esm2_t33_650M_UR50D.pt` (2.6 GB) and its `-contact-regression.pt`, the `facebookresearch_esm_v2.0.0` checkout and `trusted_list`. Source: https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt and github facebookresearch/esm v2.0.0. A read-only copy exists in `emulator_old_data/data/processed/baselines/Clipzyme/emulator_bench/cache/torch_hub`.
- Env: the `clipzyme` env is built (commands.txt). `mldb` is not used.
- Scale: the grouped split has about 1.1M train rows (72k unique reactions, 207k sequences) with ESM-650M for 20 epochs. This is many days per run on one GPU. `checkpoint_last.pt` is about 8 GB per run.
- Open choices: 8 vs 32 negatives, candidate pool, length filtering of test queries, EC ranking method, k, the epoch scheduler quirk, lr and wd.
- Frozen ESM-2 (`--frozen_esm`) follows `configs/train/clip_esm_frozen.json`: ESM-2 frozen (user decision), 10 epochs, no weight decay (not set there, code default), 32 samples per optimizer step. The authors ran 4 GPUs x 8 samples with the gathered features as constants (`concat_all_gather` has no gradient). One GPU loads the 32 samples, encodes the reactions in 4 virtual ranks of 8, and `virtual_ranks.py` computes the loss of each rank against all 32 constants, then averages them (the DDP gradient average). `from_checkpoint` and `cache_path` are not used. The ESM-2 vectors come from a fp32 cache (`esm_cache.py`); the model state then holds no ESM-2 weights.
- Frozen-path equivalence (`python -m emulator_bench.check_equivalence`, 42 steps, 4 virtual ranks, batch 32, deterministic GPU, fp32): the original code run twice is bit-identical. Original vs cache path: same initial weights, same batch draws at every step, step-0 loss differs by 6.9e-8 (relative), first-step gradients by at most 1.6e-5 (relative). The loss differs by less than 1e-5 up to step 9, and by at most 1.1e-3 over 42 steps; final weights differ by at most 3.1e-3. The cached fp32 vectors differ from the in-loop ESM-2 by about 1e-6 (padding, chunking, SDPA), and AdamW amplifies this. A control run of the original code with 1e-6 relative noise on the ESM-2 vectors diverges by the same amount (loss 1.8e-3, weights 3.6e-3). `tests/test_frozen_esm.py` asserts that cached vectors equal the in-loop ESM-2 650M vectors within 1e-5 (absolute) on real sequences of different lengths, and that the step-0 loss (relative) and gradients (absolute) match within 1e-5.
