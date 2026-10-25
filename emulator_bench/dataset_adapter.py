from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .atom_maps import (
    DEFAULT_MAP_FILE,
    clipzyme_reaction,
    load_atom_maps,
    validate_mapped_reaction,
)
from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    atomic_write,
    load_hparams,
    sha256_text,
    split_dir,
    write_json,
)

REQUIRED_COLUMNS = {"rxn_smiles", "ec_number", "sequence"}
SPLITS = ("train", "val", "test")
CLIPZYME_SPLIT = {"train": "train", "val": "dev"}
EC_PATTERN = r"\d+\.\d+\.\d+\.\d+"
PROTEIN_FOLDER = "with_protein_information"
INPUTS_VERSION = 1  # raise it when the code writes different inputs from the same files


@dataclass(frozen=True)
class Split:
    group: str
    name: str
    path: Path

    @property
    def id(self) -> str:
        return f"{self.group}/{self.name}"


def discover_splits(dataset_root: str | Path) -> tuple[list[Split], list[str]]:
    """Find every folder with train/val/test parquet files. Return (usable splits, skipped folders)."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    splits, skipped = {}, []
    for train_file in sorted(root.rglob("train.parquet")):
        folder = train_file.parent
        if any(p.startswith("backup") for p in folder.relative_to(root).parts):
            continue
        if not all((folder / f"{s}.parquet").exists() for s in SPLITS):
            continue
        if not REQUIRED_COLUMNS.issubset(pq.read_schema(train_file).names):
            skipped.append(folder.relative_to(root).as_posix())
            continue
        parts = [p for p in folder.relative_to(root).parts if p != PROTEIN_FOLDER]
        if parts[-1].startswith("threshold_"):
            group, name = "/".join(parts[:-1]), parts[-1]
        else:
            group, name = "/".join(parts), "all"
        split = Split(group, name, folder)
        if split.id in splits:
            raise ValueError(
                f"Two folders map to {split.id}: {splits[split.id].path} and {folder}"
            )
        splits[split.id] = split
    if not splits:
        raise FileNotFoundError(f"No usable train/val/test split folders under {root}")
    return list(splits.values()), skipped


def select_splits(splits: list[Split], wanted: list[str] | None) -> list[Split]:
    if not wanted:
        return splits
    chosen = [s for s in splits if s.group in wanted or s.id in wanted]
    if not chosen:
        raise ValueError(f"No split matches {wanted}. Found: {[s.id for s in splits]}")
    return chosen


def read_rows(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one split file and drop rows the original pipeline cannot use. Count each drop."""
    frame = pd.read_parquet(path, columns=sorted(REQUIRED_COLUMNS)).dropna()
    stats = {"raw_rows": int(pq.ParquetFile(path).metadata.num_rows)}
    stats["missing_value_rows"] = stats["raw_rows"] - len(frame)
    frame = frame.astype(str)
    frame["rxn_smiles"] = frame["rxn_smiles"].str.strip()
    frame["ec_number"] = frame["ec_number"].str.strip()
    frame["sequence"] = frame["sequence"].str.replace(r"\s+", "", regex=True).str.upper()
    for key, keep in (
        ("invalid_reaction_rows", frame["rxn_smiles"].str.fullmatch(r"[^>\s]+>>[^>\s]+")),
        ("incomplete_ec_rows", frame["ec_number"].str.fullmatch(EC_PATTERN)),
        ("empty_sequence_rows", frame["sequence"] != ""),
    ):
        stats[key] = int((~keep).sum())
        frame = frame[keep]
    unique = frame.drop_duplicates().reset_index(drop=True)
    stats["exact_duplicate_rows"] = len(frame) - len(unique)
    return unique, stats


def candidate_pool(frames: dict[str, pd.DataFrame], max_length: int) -> pd.DataFrame:
    """(sequence, EC) pairs of every protein of the split: train, val and test."""
    pool = pd.concat(frames.values())[["sequence", "ec_number"]].drop_duplicates()
    return pool[pool["sequence"].str.len() <= max_length].sort_values(["sequence", "ec_number"])


def file_identity(path: Path) -> list:
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def inputs_fingerprint(split: Split, atom_map_file: str | Path, limits: list[int] | None) -> dict:
    """Everything the built inputs depend on: source files, atom maps, row limits and code version."""
    return {
        "version": INPUTS_VERSION,
        "limits": limits,
        "atom_map_file": file_identity(Path(atom_map_file)),
        "sources": [file_identity(split.path / f"{name}.parquet") for name in SPLITS],
    }


def build_split_inputs(
    split: Split,
    out_root: str | Path,
    atom_maps: dict[str, str],
    atom_map_file: str | Path,
    limits: list[int] | None = None,
    force: bool = False,
) -> dict:
    """Write the Clipzyme JSON, ec2uniprot, uniprot2sequence, manifests and stats for one split."""
    inputs = split_dir(out_root, split.group, split.name) / "inputs"
    stats_path = inputs / "stats.json"
    fingerprint = inputs_fingerprint(split, atom_map_file, limits)
    if (
        stats_path.exists()
        and not force
        and json.loads(stats_path.read_text()).get("fingerprint") == fingerprint
    ):
        return json.loads(stats_path.read_text())

    frames, stats = {}, {}
    for name in SPLITS:
        frames[name], stats[name] = read_rows(split.path / f"{name}.parquet")

    featurized: dict[str, tuple] = {}
    for i, name in enumerate(SPLITS):
        frame = frames[name].assign(raw_map=frames[name]["rxn_smiles"].map(atom_maps))
        stats[name]["no_atom_map_rows"] = int(frame["raw_map"].isna().sum())
        if name != "test":
            frame = frame[frame["raw_map"].notna()]
        if limits:
            frame = frame.sample(min(limits[i], len(frame)), random_state=0)
        for raw in frame["raw_map"].dropna().unique():
            if raw not in featurized:
                featurized[raw] = validate_mapped_reaction(raw)
        frame = frame.assign(
            mapped=frame["raw_map"].map(lambda r: featurized[r][0] if r in featurized else None),
            bond_changes=frame["raw_map"].map(
                lambda r: featurized[r][1] if r in featurized else []
            ),
        )
        stats[name]["failed_featurization_rows"] = int(
            frame["raw_map"].notna().sum() - frame["mapped"].notna().sum()
        )
        if name == "test":
            # Test rows that cannot be featurized stay in the test set and are scored as misses.
            stats[name]["scored_as_miss_rows"] = int(frame["mapped"].isna().sum())
        else:
            frame = frame[frame["mapped"].notna()]
        frame = frame.reset_index(drop=True)
        if frame["mapped"].notna().sum() == 0 and name != "test":
            raise ValueError(
                f"{split.id}/{name} has no usable rows. {stats[name]['no_atom_map_rows']} rows have no atom map "
                f"in {atom_map_file}."
            )
        stats[name]["usable_rows"] = int(frame["mapped"].notna().sum())
        frames[name] = frame

    max_length = load_hparams()["max_protein_length"]
    entries, uniprot2sequence, ec2uniprot = [], {}, {}
    for name, key in CLIPZYME_SPLIT.items():
        for row in frames[name].itertuples(index=False):
            reactants, products, _ = clipzyme_reaction(row.raw_map)
            protein = f"seq_{sha256_text(row.sequence)[:20]}"
            uniprot2sequence[protein] = row.sequence
            ec2uniprot.setdefault(row.ec_number, set()).add(protein)
            raw_reactants, raw_products = row.rxn_smiles.split(">>")
            entries.append(
                {
                    "rxnid": f"rxn_{sha256_text(f'{row.rxn_smiles}|{row.sequence}|{row.ec_number}')[:24]}",
                    "quality": 1.0,
                    "ec": row.ec_number,
                    "reactants": raw_reactants.split("."),
                    "products": raw_products.split("."),
                    "mapped_reactants": row.raw_map.split(">>")[0].split("."),
                    "mapped_products": row.raw_map.split(">>")[1].split("."),
                    "protein_refs": repr([protein]),
                    "protein_db": "uniprot",
                    "rule_id": f"rule_{sha256_text(row.rxn_smiles)[:20]}",
                    "split": key,
                    "bond_changes": row.bond_changes,
                }
            )

    candidates = candidate_pool(frames, max_length)
    atomic_write(inputs / "dataset.json", lambda tmp: json.dump(entries, open(tmp, "w")))
    atomic_write(
        inputs / "uniprot2sequence.p", lambda tmp: pickle.dump(uniprot2sequence, open(tmp, "wb"))
    )
    atomic_write(
        inputs / "ec2uniprot.p",
        lambda tmp: pickle.dump({ec: sorted(p) for ec, p in ec2uniprot.items()}, open(tmp, "wb")),
    )
    atomic_write(
        inputs / "train_candidates.parquet", lambda tmp: candidates.to_parquet(tmp, index=False)
    )
    for name in ("val", "test"):
        queries = frames[name][["rxn_smiles", "mapped", "ec_number"]].drop_duplicates()
        atomic_write(
            inputs / f"{name}_queries.parquet",
            lambda tmp, q=queries: q.to_parquet(tmp, index=False),
        )
    stats = {
        "split": split.id,
        "source": str(split.path),
        "atom_map_file": str(atom_map_file),
        "limits": limits,
        "fingerprint": fingerprint,
        "splits": stats,
        "dataset_json_entries": len(entries),
        "candidate_proteins": int(candidates["sequence"].nunique()),
    }
    write_json(stats_path, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Clipzyme inputs from EMULaToR retrieval splits"
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--split_groups",
        nargs="*",
        help="Split group or group/name; default is every split on disk",
    )
    parser.add_argument(
        "--atom_map_file", default=str(DEFAULT_MAP_FILE), help="EMULaToR LocalMapper atom maps"
    )
    parser.add_argument(
        "--limit_rows", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"), help="Smoke test only"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    atom_maps = load_atom_maps(args.atom_map_file)
    splits, skipped = discover_splits(args.dataset_root)
    for folder in skipped:
        print(f"[prepare] skipped {folder}: no sequence column", flush=True)
    for split in select_splits(splits, args.split_groups):
        stats = build_split_inputs(
            split, args.out_root, atom_maps, args.atom_map_file, args.limit_rows, args.force
        )
        print(f"[prepare] {split.id}: {json.dumps(stats['splits'])}", flush=True)


if __name__ == "__main__":
    main()
