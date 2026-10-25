"""Check that the frozen-ESM cache path only speeds up the original frozen path.

Reference: the original training code with clip_freeze_esm (ESM-2 forward at every step, no cache).
Candidate: the same run with the ESM cache. Both use the frozen_hparams of default_hparams.json (4 virtual GPUs), the same seed
and the same data subset. Two settings run: fp32 on both sides (no autocast), and the production bf16 autocast.
The report lists per-step sample ids, losses, final trainable weights and, for bf16, the EC ranking of the test reactions.

    python -m emulator_bench.check_equivalence --work_dir DIR
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pickle
import runpy
import sys
from argparse import Namespace
from pathlib import Path

RECORD_ENV = "EQUIVALENCE_RECORD"


def run_recorded() -> None:
    """Entry point of a training subprocess: run_main plus a record of sample ids, losses and initial weights."""
    import torch
    from clipzyme.lightning.base import Base

    if os.environ.get("EQUIVALENCE_MAX_GB"):
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ["EQUIVALENCE_MAX_GB"])
            * 2**30
            / torch.cuda.get_device_properties(0).total_memory
        )

    if os.environ.get("EQUIVALENCE_DETERMINISTIC"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if os.environ.get("EQUIVALENCE_ESM_NOISE"):
        from clipzyme.models.fair_esm import FairEsm2

        esm_forward = FairEsm2.forward
        noise = torch.Generator(device="cuda").manual_seed(0)

        def noisy_forward(self, x):
            out = esm_forward(self, x)
            hidden = out["hidden"]
            scale = float(os.environ["EQUIVALENCE_ESM_NOISE"]) * hidden.abs().amax(1, keepdim=True)
            out["hidden"] = hidden + scale * torch.randn(
                hidden.shape, device=hidden.device, generator=noise
            )
            return out

        FairEsm2.forward = noisy_forward

    record = Path(os.environ[RECORD_ENV])
    steps = []
    training_step, fit_start = Base.training_step, Base.on_fit_start
    saved_grads = []

    def trainable(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.named_parameters() if v.requires_grad}

    def on_fit_start(self):
        import torch

        torch.save(trainable(self), record.with_suffix(".init.pt"))
        fit_start(self)

    def step(self, batch, batch_idx):
        output = training_step(self, batch, batch_idx)
        steps.append({"ids": list(batch["sample_id"]), "loss": float(output["loss"])})
        return output

    def on_after_backward(self):
        if not saved_grads:  # gradients of the first step, before any update
            saved_grads.append(1)
            torch.save(
                {
                    k: v.grad.detach().cpu().clone()
                    for k, v in self.named_parameters()
                    if v.grad is not None
                },
                record.with_suffix(".grad.pt"),
            )

    Base.training_step, Base.on_fit_start = step, on_fit_start
    Base.on_after_backward = on_after_backward
    atexit.register(lambda: record.write_text(json.dumps(steps)))
    sys.argv = ["run_main", *sys.argv[2:]]
    runpy.run_module("emulator_bench.run_main", run_name="__main__")


def set_option(command: list[str], key: str, value: str) -> None:
    command[command.index(f"--{key}") + 1] = value


def run_side(args, name: str, tag: str, precision: str, work: Path, group: str, split: str):
    """Train one side with the wrapper's train() and return its run folder."""
    from . import train as train_module

    out_root = work / f"{name}_{tag}"
    run_args = Namespace(
        out_root=str(out_root),
        dataset_root=args.dataset_root,
        seed=args.seed,
        amp=False,
        checkpoint_esm=False,
        fast=False,
        frozen_esm=True,
        no_esm_cache=tag != "cand",
        esm_cache_dir=str(work / "cache"),
        hub_dir=args.hub_dir,
        epochs=args.epochs,
        num_workers=args.num_workers,
    )
    original = train_module.build_command

    def build(a, g, n):
        command = original(a, g, n)
        command[1:3] = ["-m", "emulator_bench.check_equivalence"]
        command.insert(3, "record")
        set_option(command, "precision", precision)
        set_option(command, "max_protein_length", str(args.max_length))
        command += [
            "--max_reactant_size",
            str(args.max_atoms),
            "--max_product_size",
            str(args.max_atoms),
        ]
        set_option(
            command, "warmup_start_lr", "0.0001"
        )  # the default warm-up lr is near 0 for these few epochs
        return command

    train_module.build_command = build
    os.environ[RECORD_ENV] = str(work / f"{name}_{tag}.json")
    os.environ.pop("EQUIVALENCE_ESM_NOISE", None)
    if tag == "noise":  # the original code with fp32-sized noise on the ESM vectors
        os.environ["EQUIVALENCE_ESM_NOISE"] = "1e-6"
    try:
        return train_module.train(run_args, group, split)
    finally:
        train_module.build_command = original


def compare_runs(ref: Path, cand: Path, ref_record: Path, cand_record: Path) -> dict:
    import torch

    ref_steps, cand_steps = json.loads(ref_record.read_text()), json.loads(cand_record.read_text())
    same_batches = [a["ids"] for a in ref_steps] == [b["ids"] for b in cand_steps]
    losses = torch.tensor(
        [[a["loss"], b["loss"]] for a, b in zip(ref_steps, cand_steps)], dtype=torch.float64
    )
    loss_abs = (losses[:, 0] - losses[:, 1]).abs()
    ref_w = torch.load(ref / "checkpoint_last.pt", map_location="cpu")["state_dict"]
    cand_w = torch.load(cand / "checkpoint_last.pt", map_location="cpu")["state_dict"]
    init = torch.load(ref_record.with_suffix(".init.pt"))
    keys = [k for k in ref_w if not k.startswith("model.protein_encoder.model.")]
    assert all(k in cand_w for k in keys)
    weight_abs = max((ref_w[k] - cand_w[k]).abs().max().item() for k in keys)
    weight_rel = max(
        ((ref_w[k] - cand_w[k]).abs().max() / ref_w[k].abs().max().clamp_min(1e-12)).item()
        for k in keys
    )
    moved = max((ref_w[k] - init[k]).abs().max().item() for k in keys if k in init)
    cand_init = torch.load(cand_record.with_suffix(".init.pt"))
    ref_grad = torch.load(ref_record.with_suffix(".grad.pt"))
    cand_grad = torch.load(cand_record.with_suffix(".grad.pt"))
    grad_rel = {
        k: (
            (ref_grad[k] - cand_grad[k]).abs().max() / ref_grad[k].abs().max().clamp_min(1e-30)
        ).item()
        for k in ref_grad
    }
    first_bad = next(
        (i for i, (a, b) in enumerate(zip(ref_steps, cand_steps)) if a["ids"] != b["ids"]), None
    )
    return {
        "steps": len(ref_steps),
        "batch_draws_identical": same_batches,
        "first_different_batch": first_bad,
        "loss_max_abs": loss_abs.max().item(),
        "loss_max_rel": (loss_abs / losses[:, 0].abs()).max().item(),
        "loss_rel_by_step": (loss_abs / losses[:, 0].abs()).tolist(),
        "initial_weights_max_abs": max((init[k] - cand_init[k]).abs().max().item() for k in init),
        "first_step_grad_max_rel": max(grad_rel.values()),
        "first_step_grad_worst_param": max(grad_rel, key=grad_rel.get),
        "first_loss_ref_cand": losses[0].tolist(),
        "last_loss_ref_cand": losses[-1].tolist(),
        "weights_max_abs": weight_abs,
        "weights_max_rel": weight_rel,
        "weights_max_abs_change_from_init": moved,
    }


def compare_rankings(args, work: Path, group: str, split: str) -> dict:
    """Rank the test reactions with both bf16 models and compare."""
    import pandas as pd

    from . import evaluate as evaluate_module

    summaries, ranked = {}, {}
    for side, cached in (("ref", False), ("cand", True)):
        out_root = work / f"prod_{side}"
        eval_args = Namespace(
            out_root=str(out_root),
            seed=args.seed,
            checkpoint=None,
            device=None,
            eval_batch_size=8,
            amp=False,
            fast=False,
            frozen_esm=True,
            no_esm_cache=not cached,
            esm_cache_dir=str(work / "cache"),
            hub_dir=args.hub_dir,
        )
        summaries[side] = evaluate_module.evaluate(eval_args, group, split)["accuracy"]
        run = out_root / "enzyme_retrieval" / group / split / f"seed_{args.seed}"
        ranked[side] = pd.read_parquet(run / "predictions.parquet")
    a, b = ranked["ref"]["ranked_ecs"], ranked["cand"]["ranked_ecs"]
    top = {k: sum(list(x[:k]) == list(y[:k]) for x, y in zip(a, b)) / len(a) for k in (1, 10)}
    return {
        "accuracy_ref": summaries["ref"],
        "accuracy_cand": summaries["cand"],
        "same_top_k_fraction": top,
    }


def build_cache(args, sequences: list[str], work: Path) -> None:
    from . import esm_cache

    model, alphabet = esm_cache.load_esm(args.hub_dir, "cuda", sdpa=True)
    esm_cache.build_cache(
        sequences,
        work / "cache",
        lambda seqs: esm_cache.pool_sequences(
            model, alphabet, seqs, "cuda", budget=args.chunk_tokens
        ),
        esm_cache.fingerprint(args.hub_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    from .atom_maps import DEFAULT_MAP_FILE
    from .utils import DEFAULT_DATASET_ROOT, DEFAULT_OUT_ROOT

    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--atom_map_file", default=str(DEFAULT_MAP_FILE))
    parser.add_argument("--hub_dir", default=str(DEFAULT_OUT_ROOT / "weights" / "torch_hub"))
    parser.add_argument("--split", default="random_splits_grouped_rxn_key/all")
    parser.add_argument("--limit_rows", type=int, nargs=3, default=[1200, 200, 300])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic GPU kernels in the training runs (warn when an op has none)",
    )
    parser.add_argument("--control", action="store_true", help="Also run the reference twice")
    parser.add_argument("--max_gb", type=float, default=None, help="GPU memory cap per process")
    parser.add_argument(
        "--max_length",
        type=int,
        default=400,
        help="Longest training protein; keeps the reference ESM-2 batches small",
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=25,
        help="Largest molecule of a training reaction; keeps the dense edge tensors small",
    )
    parser.add_argument("--chunk_tokens", type=int, default=4096)
    parser.add_argument(
        "--settings", nargs="+", default=["fp32", "prod"], choices=["fp32", "prod"]
    )
    args = parser.parse_args()

    import pandas as pd
    import torch

    if args.deterministic:
        os.environ["EQUIVALENCE_DETERMINISTIC"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.max_gb:
        os.environ["EQUIVALENCE_MAX_GB"] = str(args.max_gb)
        torch.cuda.set_per_process_memory_fraction(
            args.max_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory
        )
    from .atom_maps import load_atom_maps
    from .dataset_adapter import build_split_inputs, discover_splits, select_splits
    from .utils import split_dir

    work = Path(args.work_dir)
    split = select_splits(discover_splits(args.dataset_root)[0], [args.split])[0]
    maps = load_atom_maps(args.atom_map_file)
    sequences = set()
    for name in args.settings:
        for side in ("ref", "cand", "ref2", "noise"):
            stats = build_split_inputs(
                split, work / f"{name}_{side}", maps, args.atom_map_file, args.limit_rows
            )
            inputs = split_dir(work / f"{name}_{side}", split.group, split.name) / "inputs"
            sequences.update(pickle.load(open(inputs / "uniprot2sequence.p", "rb")).values())
            sequences.update(pd.read_parquet(inputs / "train_candidates.parquet")["sequence"])
    sequences = sorted(s for s in sequences if len(s) <= 650)
    print(f"usable rows {stats['splits']}; {len(sequences)} sequences", flush=True)
    build_cache(args, sequences, work)
    torch.cuda.empty_cache()

    report = {}
    for name in args.settings:
        precision = "32-true" if name == "fp32" else "bf16"
        tags = ("ref", "cand", "ref2", "noise") if args.control else ("ref", "cand")
        for tag in tags:
            run_side(args, name, tag, precision, work, split.group, split.name)
        runs = {
            side: work
            / f"{name}_{side}"
            / "enzyme_retrieval"
            / split.group
            / split.name
            / f"seed_{args.seed}"
            for side in tags
        }
        report[name] = compare_runs(
            runs["ref"], runs["cand"], work / f"{name}_ref.json", work / f"{name}_cand.json"
        )
        if args.control:  # the same original code twice: the noise of the GPU itself
            report[name]["control_ref_vs_ref"] = compare_runs(
                runs["ref"], runs["ref2"], work / f"{name}_ref.json", work / f"{name}_ref2.json"
            )
            report[name]["control_ref_vs_ref_with_1e-6_noise_on_esm_vectors"] = compare_runs(
                runs["ref"], runs["noise"], work / f"{name}_ref.json", work / f"{name}_noise.json"
            )
    if "prod" in args.settings:
        report["prod"]["ranking"] = compare_rankings(args, work, split.group, split.name)
    print(json.dumps(report, indent=2), flush=True)
    (work / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"]:
        run_recorded()
    else:
        main()
