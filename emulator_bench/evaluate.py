from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_adapter import discover_splits, select_splits
from .results import K_VALUES, ec_level_accuracy
from . import esm_cache
from .train import add_run_args, esm_cache_dir, hub_dir
from .utils import atomic_write, read_json, seed_dir, split_dir, write_json

# Checkpoints hold argparse.Namespace objects, so they need the full pickle loader.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

TOP_K = max(K_VALUES)


def load_model(checkpoint: Path, device: str):
    from argparse import Namespace

    from clipzyme.lightning.clipzyme import CLIPZyme

    wrapper = CLIPZyme(
        args=Namespace(
            checkpoint_path=str(checkpoint),
            use_as_protein_encoder=False,
            use_as_reaction_encoder=False,
        )
    )
    wrapper.model.to(device).eval()
    return wrapper


def encode(
    items: list,
    collate_one,
    encoder,
    batch_size: int,
    device: str,
    amp: bool,
    by_length: bool = False,
):
    """Embed items in batches with a CLIPZyme feature extractor. Embeddings stay fp32.

    With by_length, items are batched in order of length and the embeddings return in the input order.
    """
    import torch
    from clipzyme.utils.loading import default_collate

    order = (
        sorted(range(len(items)), key=lambda i: len(items[i])) if by_length else range(len(items))
    )
    items = [items[i] for i in order]
    out = []
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp),
    ):
        for start in range(0, len(items), batch_size):
            batch = default_collate(
                [collate_one(item) for item in items[start : start + batch_size]]
            )
            batch = {
                key: val.to(device) if hasattr(val, "to") else val for key, val in batch.items()
            }
            out.append(encoder(batch).float())
    out = torch.cat(out)
    if by_length:
        restored = torch.empty_like(out)
        restored[torch.tensor(order, device=out.device)] = out
        return restored
    return out


def rank_ecs(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    wrapper,
    device: str,
    batch_size: int,
    amp: bool,
    fast: bool = False,
):
    """Rank the pool ECs for each test reaction by the best cosine to any pool enzyme with that EC."""
    import torch
    from clipzyme.utils.screening import process_mapped_reaction

    sequences = sorted(candidates["sequence"].unique())
    seq_index = {s: i for i, s in enumerate(sequences)}
    ecs = sorted(candidates["ec_number"].unique())
    ec_index = {e: i for i, e in enumerate(ecs)}
    pair_protein = torch.tensor(candidates["sequence"].map(seq_index).to_numpy(), device=device)
    pair_ec = torch.tensor(candidates["ec_number"].map(ec_index).to_numpy(), device=device)

    proteins = encode(
        sequences,
        lambda s: {"sequence": s, "sample_id": s},
        wrapper.extract_protein_features,
        batch_size,
        device,
        amp,
        by_length=fast,
    )

    def reaction_item(mapped: str) -> dict:
        reactants, products = process_mapped_reaction(mapped)
        return {"reactants": reactants, "products": products, "sample_id": mapped}

    reactions = encode(
        list(queries["mapped"]),
        reaction_item,
        wrapper.extract_reaction_features,
        batch_size,
        device,
        amp,
    )
    top_scores, top_ecs = [], []
    k = min(TOP_K, len(ecs))
    for start in range(0, len(reactions), 128):
        scores = reactions[start : start + 128] @ proteins.T
        ec_scores = torch.full((len(scores), len(ecs)), -torch.inf, device=device)
        ec_scores.scatter_reduce_(
            1, pair_ec.expand(len(scores), -1), scores[:, pair_protein], "amax"
        )
        values, indices = ec_scores.topk(k, dim=1)
        top_scores.append(values.cpu())
        top_ecs.append(indices.cpu())
    top_scores, top_ecs = torch.cat(top_scores), torch.cat(top_ecs)
    ranked = [[ecs[i] for i in row] for row in top_ecs.tolist()]
    return ranked, top_scores.numpy(), len(sequences), len(ecs)


def evaluate(args: argparse.Namespace, group: str, name: str) -> dict:
    import torch

    run = seed_dir(args.out_root, group, name, args.seed)
    inputs = split_dir(args.out_root, group, name) / "inputs"
    checkpoint = Path(args.checkpoint) if args.checkpoint else run / "bestmodel.pth"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.fast:
        from . import fast_esm

        fast_esm.patch_esm_kernels()
    candidates = pd.read_parquet(inputs / "train_candidates.parquet")
    if args.frozen_esm and not args.no_esm_cache:
        cache = esm_cache_dir(args)
        esm_cache.check_cache(
            cache, esm_cache.fingerprint(hub_dir(args)), list(candidates["sequence"])
        )
        esm_cache.install_frozen_esm(cache)
    wrapper = load_model(checkpoint, device)

    queries = pd.read_parquet(inputs / "test_queries.parquet")
    truth = queries.groupby("rxn_smiles", sort=True).agg(
        mapped=("mapped", "first"), true_ecs=("ec_number", lambda x: sorted(set(x)))
    )
    truth = truth.reset_index()
    usable = truth["mapped"].notna()
    ranked_usable, scores_usable, n_proteins, n_ecs = rank_ecs(
        truth[usable], candidates, wrapper, device, args.eval_batch_size, args.amp, args.fast
    )
    # Test reactions without a usable atom map get an empty ranking, so every top-k counts them as a miss.
    ranked, scores = [[]] * len(truth), [np.empty(0, dtype=np.float32)] * len(truth)
    for row, r, sc in zip(np.flatnonzero(usable), ranked_usable, scores_usable):
        ranked[row], scores[row] = r, sc

    predictions = pd.DataFrame(
        {
            "rxn_smiles": truth["rxn_smiles"],
            "true_ecs": truth["true_ecs"],
            "ranked_ecs": ranked,
            "ranked_scores": scores,
        }
    )
    atomic_write(run / "predictions.parquet", lambda tmp: predictions.to_parquet(tmp, index=False))
    accuracy = ec_level_accuracy(predictions)
    summary = {
        "split": f"{group}/{name}",
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "queries": len(predictions),
        "scored_as_miss_queries": int((~usable).sum()),
        "candidate_proteins": n_proteins,
        "candidate_ecs": n_ecs,
        "accuracy": accuracy,
    }
    write_json(run / "metrics.json", summary)
    run_summary = {
        **summary,
        "prepare": read_json(inputs / "stats.json")["splits"],
        "train": read_json(run / "train_summary.json")
        if (run / "train_summary.json").exists()
        else None,
    }
    write_json(run / "run_summary.json", run_summary)
    print(f"[evaluate] {group}/{name} seed {args.seed}: {accuracy['k=1']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank enzymes for the test reactions and write EC-level accuracy"
    )
    add_run_args(parser)
    parser.add_argument("--checkpoint", default=None, help="Default: <run>/bestmodel.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()
    seeds = args.seeds
    for split in select_splits(discover_splits(args.dataset_root)[0], args.split_groups):
        for seed in seeds:
            args.seed = seed
            evaluate(args, split.group, split.name)


if __name__ == "__main__":
    main()
