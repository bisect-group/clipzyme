from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "datasets" / "enzyme_retrieval_dataset"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "Clipzyme"
TASK = "enzyme_retrieval"
DEFAULT_SEEDS = (666, 777, 888)
HPARAMS_PATH = Path(__file__).with_name("default_hparams.json")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def atomic_write(path: str | Path, write: Callable[[str], None]) -> None:
    """Call write(tmp_path) with a temp file inside the target folder, then rename it onto path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: str | Path, data: object) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    atomic_write(path, lambda tmp: Path(tmp).write_text(text))


def load_hparams(frozen_esm: bool = False) -> dict:
    """The default hparams. With frozen_esm, the frozen_hparams section replaces the keys it lists."""
    data = read_json(HPARAMS_PATH)
    return {**data["hparams"], **(data["frozen_hparams"] if frozen_esm else {})}


def split_dir(out_root: str | Path, group: str, name: str) -> Path:
    return Path(out_root) / TASK / group / name


def seed_dir(out_root: str | Path, group: str, name: str, seed: int) -> Path:
    return split_dir(out_root, group, name) / f"seed_{seed}"
