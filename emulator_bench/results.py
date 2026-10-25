from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .utils import DEFAULT_OUT_ROOT, TASK, atomic_write

K_VALUES = (1, 3, 5, 10, 20, 30, 40, 50)
LEVELS = (4, 3, 2, 1)


def matched_levels(true_ec: str, ranked: list[str]) -> int:
    """Longest EC prefix (0-4 levels) that the true EC shares with any ranked EC."""
    true_parts = true_ec.split(".")
    best = 0
    for ec in ranked:
        n = 0
        for a, b in zip(ec.split("."), true_parts):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


def ec_level_accuracy(predictions: pd.DataFrame, k_values=K_VALUES) -> dict[str, dict[str, float]]:
    """Level-L accuracy at k in percent: share of true ECs matched to level L by the top-k ranked ECs.

    A reaction with several true ECs counts each EC with equal weight inside that reaction.
    """
    metrics = {}
    for k in k_values:
        hits = {level: [] for level in LEVELS}
        for true_ecs, ranked in zip(predictions["true_ecs"], predictions["ranked_ecs"]):
            matched = [matched_levels(ec, list(ranked[:k])) for ec in true_ecs]
            for level in LEVELS:
                hits[level].append(sum(m >= level for m in matched) / len(matched))
        metrics[f"k={k}"] = {f"level_{level}": 100 * sum(h) / len(h) for level, h in hits.items()}
    return metrics


def aggregate(out_root: str | Path) -> None:
    """Rebuild the summary CSVs from the saved predictions of every finished run."""
    root = Path(out_root) / TASK
    rows = []
    for path in sorted(root.glob("**/seed_*/predictions.parquet")):
        group_name, seed = (
            path.parent.parent.relative_to(root),
            int(path.parent.name.split("_")[1]),
        )
        predictions = pd.read_parquet(path)
        metrics = ec_level_accuracy(predictions)
        saved = path.parent / "metrics.json"
        if saved.exists() and json.loads(saved.read_text())["accuracy"] != json.loads(
            json.dumps(metrics)
        ):
            print(
                f"[results] metrics.json differs from the predictions: {path.parent}", flush=True
            )
        for k, levels in metrics.items():
            rows.append(
                {
                    "split_group": "/".join(group_name.parts[:-1]),
                    "split_name": group_name.name,
                    "seed": seed,
                    "k": int(re.sub(r"\D", "", k)),
                    "queries": len(predictions),
                    "scored_as_miss": int((predictions["ranked_ecs"].map(len) == 0).sum()),
                    **levels,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No predictions.parquet under {root}")
    by_seed = pd.DataFrame(rows)
    keys = ["split_group", "split_name", "k"]
    summary = by_seed.groupby(keys)[[f"level_{level}" for level in LEVELS]].agg(["mean", "std"])
    summary.columns = [f"{level}_{stat}" for level, stat in summary.columns]
    summary["seeds"] = by_seed.groupby(keys)["seed"].nunique()
    atomic_write(root / "summary_by_seed.csv", lambda tmp: by_seed.to_csv(tmp, index=False))
    atomic_write(root / "summary.csv", lambda tmp: summary.reset_index().to_csv(tmp, index=False))
    print(f"[results] wrote {root / 'summary.csv'} ({len(by_seed)} rows)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild Clipzyme retrieval summaries from saved predictions"
    )
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    aggregate(parser.parse_args().out_root)
