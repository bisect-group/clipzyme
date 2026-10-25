from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .atom_maps import DEFAULT_MAP_FILE
from .dataset_adapter import discover_splits, select_splits
from .utils import (
    BASELINE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    TASK,
    seed_dir,
)


def run_logged(command: list[str], log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        return subprocess.run(
            command, cwd=BASELINE_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Clipzyme on every retrieval split and seed"
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--split_groups",
        nargs="*",
        help="Split group or group/name; default is every split on disk",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="GPU ids; each run gets one through CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bf16 on Ampere or newer, fp16 on older GPUs, fp32 on CPU",
    )
    parser.add_argument(
        "--hub_dir",
        default=None,
        help="torch hub dir with ESM-2 weights; default <out_root>/weights/torch_hub",
    )
    parser.add_argument(
        "--atom_map_file", default=str(DEFAULT_MAP_FILE), help="EMULaToR LocalMapper atom maps"
    )
    parser.add_argument(
        "--checkpoint_esm",
        action="store_true",
        help="Recompute ESM-2 activations; needed for batch 8 on 23 GB",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Exact speed-ups, see emulator_bench/fast_esm.py"
    )
    parser.add_argument(
        "--frozen_esm",
        action="store_true",
        help="Frozen ESM-2 from the ESM cache, see emulator_bench/esm_cache.py",
    )
    parser.add_argument("--esm_cache_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true", help="List the runs and stop")
    args = parser.parse_args()

    splits, skipped = discover_splits(args.dataset_root)
    for folder in skipped:
        print(f"skipped {folder}: no sequence column", flush=True)
    splits = select_splits(splits, args.split_groups)
    jobs = [(s, seed) for s in splits for seed in args.seeds]
    for split, seed in jobs:
        print(
            f"run {split.id} seed {seed} -> {seed_dir(args.out_root, split.group, split.name, seed)}",
            flush=True,
        )
    if args.dry_run:
        return

    py = [sys.executable, "-m"]
    common = ["--dataset_root", args.dataset_root, "--out_root", args.out_root]
    if args.amp:
        common.append("--amp")
    if args.checkpoint_esm:
        common.append("--checkpoint_esm")
    if args.fast:
        common.append("--fast")
    if args.frozen_esm:
        common.append("--frozen_esm")
    if args.esm_cache_dir:
        common += ["--esm_cache_dir", args.esm_cache_dir]
    if args.epochs is not None:
        common += ["--epochs", str(args.epochs)]
    if args.hub_dir:
        common += ["--hub_dir", args.hub_dir]
    logs = Path(args.out_root) / "logs" / TASK
    for split in splits:  # CPU: atom maps and Clipzyme input files, once per split
        command = [*py, "emulator_bench.dataset_adapter", *common[:4], "--split_groups", split.id]
        command += ["--atom_map_file", args.atom_map_file]
        log = logs / f"{split.group}__{split.name}__prepare.log"
        if run_logged(command, log, {**os.environ, "CUDA_VISIBLE_DEVICES": ""}):
            sys.exit(f"prepare failed for {split.id}; see {logs}")

    slots: queue.Queue = queue.Queue()
    for _ in range(args.runs_per_gpu):
        for gpu in args.gpus:
            slots.put(gpu)

    def run(job) -> bool:
        split, seed = job
        gpu = slots.get()
        try:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            select = [
                "--split_groups",
                split.id,
                "--seeds",
                str(seed),
                "--num_workers",
                str(args.num_workers),
            ]
            log = logs / f"{split.group}__{split.name}__seed_{seed}.log"
            ok = all(
                run_logged([*py, f"emulator_bench.{stage}", *common, *select], log, env) == 0
                for stage in ("train", "evaluate")
            )
            print(
                f"{'done' if ok else 'FAILED'} {split.id} seed {seed} (gpu {gpu}); log {log}",
                flush=True,
            )
            return ok
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(args.gpus) * args.runs_per_gpu) as pool:
        results = list(pool.map(run, jobs))
    if not all(results):
        sys.exit(1)
    subprocess.run(
        [*py, "emulator_bench.results", "--out_root", args.out_root], cwd=BASELINE_ROOT, check=True
    )


if __name__ == "__main__":
    main()
