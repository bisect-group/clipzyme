from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from . import esm_cache
from .dataset_adapter import discover_splits, select_splits
from .utils import (
    BASELINE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    atomic_write,
    load_hparams,
    seed_dir,
    split_dir,
    write_json,
)


def choose_precision(amp: bool, default: str) -> str:
    """The original precision, or with --amp: bf16 on Ampere or newer, fp16 on older GPUs, fp32 on CPU."""
    if not amp:
        return default
    import torch

    if not torch.cuda.is_available():
        return "32-true"
    return "bf16" if torch.cuda.is_bf16_supported() else "16-mixed"


def build_command(args: argparse.Namespace, group: str, name: str) -> list[str]:
    hp = load_hparams(args.frozen_esm)
    hp.pop("virtual_gpus", None)  # not a main.py argument; see train()
    hp["precision"] = choose_precision(args.amp, hp["precision"])
    if args.epochs is not None:
        hp["max_epochs"] = args.epochs
    run = seed_dir(args.out_root, group, name, args.seed)
    inputs = split_dir(args.out_root, group, name) / "inputs"
    hp.update(
        dataset_file_path=inputs / "dataset.json",
        ec2uniprot_path=inputs / "ec2uniprot.p",
        uniprot2sequence_path=inputs / "uniprot2sequence.p",
        pretrained_hub_dir=hub_dir(args),
        checkpoint_dir=run.parent,
        experiment_name=run.name,
        results_path=run / "args",
        logger_name="tensorboard",
        logger_dir=run / "tensorboard",
        num_workers=args.num_workers,
        gpus=1,
        seed=args.seed,
    )
    argv = []
    for key, value in hp.items():
        if value is True:
            argv.append(f"--{key}")
        elif isinstance(value, list):
            argv += [f"--{key}", *map(str, value)]
        elif value is not False:
            argv += [f"--{key}", str(Path(value).resolve() if isinstance(value, Path) else value)]
    return [sys.executable, "-m", "emulator_bench.run_main", *argv]


def finalize(run: Path) -> None:
    """Write bestmodel.pth and lastmodel.pth (weights only) and train_summary.json."""
    import torch

    def weights_only(source: Path, target: Path) -> dict:
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        keep = {
            k: ckpt[k]
            for k in ("state_dict", "hyper_parameters", "epoch", "global_step")
            if k in ckpt
        }
        atomic_write(target, lambda tmp: torch.save(keep, tmp))
        return ckpt

    best_path = Path(pickle.load(open(run / "args.args", "rb"))["model_path"])
    best = weights_only(best_path, run / "bestmodel.pth")
    score = next(
        v["best_model_score"] for v in best["callbacks"].values() if "best_model_score" in v
    )
    last = weights_only(run / "checkpoint_last.pt", run / "lastmodel.pth")
    counts = json.loads((run / "native_counts.json").read_text())
    write_json(
        run / "train_summary.json",
        {
            "best_epoch": best["epoch"],
            "best_val_clip_quantile": float(score),
            "last_epoch": last["epoch"],
            "native_dataset_sizes": counts,
        },
    )
    best_path.unlink()


def esm_cache_dir(args: argparse.Namespace) -> Path:
    return Path(args.esm_cache_dir or esm_cache.cache_dir(args.out_root))


def hub_dir(args: argparse.Namespace) -> Path:
    return Path(args.hub_dir or Path(args.out_root) / "weights" / "torch_hub")


def check_esm_cache(args: argparse.Namespace, group: str, name: str) -> None:
    """Stop before training when the ESM cache lacks a protein of the split."""
    inputs = split_dir(args.out_root, group, name) / "inputs"
    sequences = pickle.load(open(inputs / "uniprot2sequence.p", "rb")).values()
    limit = load_hparams()["max_protein_length"]
    esm_cache.check_cache(
        esm_cache_dir(args),
        esm_cache.fingerprint(hub_dir(args)),
        [s for s in sequences if len(s) <= limit],
    )


def train(args: argparse.Namespace, group: str, name: str) -> Path:
    run = seed_dir(args.out_root, group, name, args.seed)
    if (run / "bestmodel.pth").exists() and (run / "lastmodel.pth").exists():
        print(f"[train] {group}/{name} seed {args.seed}: already trained", flush=True)
        return run
    run.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "EMULATOR_COUNTS_PATH": str(run / "native_counts.json")}
    if args.checkpoint_esm:
        env["EMULATOR_CHECKPOINT_ESM"] = "1"
    if args.fast:
        env["EMULATOR_FAST"] = "1"
    if args.frozen_esm:
        if not args.no_esm_cache:
            check_esm_cache(args, group, name)
            env["EMULATOR_FROZEN_ESM"] = str(esm_cache_dir(args))
        env["EMULATOR_VIRTUAL_GPUS"] = str(load_hparams(True)["virtual_gpus"])
    if (run / "checkpoint_last.pt").exists():
        env["EMULATOR_RESUME_CKPT"] = str(run / "checkpoint_last.pt")
    command = build_command(args, group, name)
    print(f"[train] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=BASELINE_ROOT, env=env, check=True)
    finalize(run)
    return run


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--split_groups",
        nargs="*",
        help="Split group or group/name; default is every split on disk",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--hub_dir",
        default=None,
        help="torch hub dir with ESM-2 weights; default <out_root>/weights/torch_hub",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bf16 on Ampere or newer, fp16 on older GPUs, fp32 on CPU",
    )
    parser.add_argument(
        "--checkpoint_esm",
        action="store_true",
        help="Recompute ESM-2 activations in the backward pass. Same numbers, less memory (batch 8 needs it on a 23 GB GPU)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Exact speed-ups: SDPA and F.gelu in ESM-2, length-sorted ESM chunks, recomputation only for large micro-batches, "
        "vectorised reverse edges, length-sorted candidate encoding",
    )
    parser.add_argument(
        "--frozen_esm",
        action="store_true",
        help="Frozen ESM-2 as in configs/train/clip_esm_frozen.json: vectors from the ESM cache, "
        "frozen_hparams of default_hparams.json (batch 32 per step emulating 4 GPUs, 10 epochs)",
    )
    parser.add_argument(
        "--no_esm_cache",
        action="store_true",
        help="With --frozen_esm: run ESM-2 at every step, as the original frozen code does (reference for check_equivalence)",
    )
    parser.add_argument(
        "--esm_cache_dir", default=None, help="Default <out_root>/embeddings/esm_frozen_v1"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Replace max_epochs (default 20, 10 with --frozen_esm)",
    )
    parser.add_argument("--num_workers", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Clipzyme on prepared EMULaToR retrieval splits"
    )
    add_run_args(parser)
    args = parser.parse_args()
    seeds = args.seeds
    for split in select_splits(discover_splits(args.dataset_root)[0], args.split_groups):
        for seed in seeds:
            args.seed = seed
            train(args, split.group, split.name)


if __name__ == "__main__":
    main()
