"""Cache of frozen ESM-2 protein vectors for CLIPZyme.

With a frozen ESM-2 the model reads one vector per protein: the mean over residues of ESM layer 33 without the
CLS and EOS tokens. ESM-2 has no dropout, so this vector is the same at every step and can be computed once.
The cache stores it in fp32 (no autocast, TF32 off), keyed by the sha256 of the sequence, in part files:

    <cache_dir>/manifest.json         fingerprint of the recipe; a different recipe is refused
    <cache_dir>/part_<hash>.pt        {"keys": [sha256, ...], "vectors": fp32 tensor [n, 1280]}

A shard owns the sequences whose key modulo shards equals its number, so shards never overlap and a stopped
run continues from the finished parts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    atomic_write,
    read_json,
    sha256_text,
    write_json,
)

if TYPE_CHECKING:
    from torch import Tensor

ESM_NAME = "esm2_t33_650M_UR50D"
LAYER = 33
RECIPE = "mean over residues of the layer output without cls, eos and padding tokens"
CHUNK_TOKENS = 24576
PART_SEQUENCES = 2048


def cache_dir(out_root: str | Path) -> Path:
    return Path(out_root) / "embeddings" / "esm_frozen_v1"


def fingerprint(hub_dir: str | Path) -> dict:
    weights = Path(hub_dir) / "checkpoints" / f"{ESM_NAME}.pt"
    return {
        "model": ESM_NAME,
        "weights_bytes": weights.stat().st_size,
        "layer": LAYER,
        "recipe": RECIPE,
        "precision": "fp32",
        "tf32": False,
    }


def pool_sequences(
    model,
    alphabet,
    sequences: list[str],
    device: str,
    layer: int = LAYER,
    budget: int = CHUNK_TOKENS,
):
    """fp32 pooled vectors [n, dim] on the CPU in input order. Sequences run in length-sorted chunks of about budget tokens."""
    import torch

    from .fast_esm import token_chunks

    converter = alphabet.get_batch_converter()
    out = None
    with torch.no_grad():
        for chunk in token_chunks([len(s) for s in sequences], budget):
            _, _, tokens = converter([(i, sequences[i]) for i in chunk])
            tokens = tokens.to(device)
            hidden = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][
                layer
            ]
            mask = (
                (
                    (tokens != alphabet.cls_idx)
                    & (tokens != alphabet.eos_idx)
                    & (tokens != alphabet.padding_idx)
                )
                .long()
                .unsqueeze(-1)
            )
            pooled = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu()
            if out is None:
                out = torch.empty(len(sequences), pooled.shape[1])
            out[chunk] = pooled
    return out


def load_esm(hub_dir: str | Path, device: str, sdpa: bool):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.hub.set_dir(str(hub_dir))
    model, alphabet = torch.hub.load("facebookresearch/esm:v2.0.0", ESM_NAME)
    if sdpa:
        from .fast_esm import patch_esm_kernels

        patch_esm_kernels()
    return model.eval().to(device), alphabet


def build_cache(
    sequences: list[str],
    out_dir: str | Path,
    embed: Callable[[list[str]], Tensor],
    recipe: dict,
    shard: int = 0,
    shards: int = 1,
    part_size: int = PART_SEQUENCES,
) -> int:
    """Embed the sequences of this shard that no part holds yet. Return the number of new vectors."""
    import torch

    out_dir = Path(out_dir)
    manifest = out_dir / "manifest.json"
    if manifest.exists() and read_json(manifest) != recipe:
        raise ValueError(f"{out_dir} was built with another recipe; use a new folder or delete it")
    write_json(manifest, recipe)
    done = set()
    for part in out_dir.glob("part_*.pt"):
        done.update(torch.load(part)["keys"])
    todo = {}
    for seq in sequences:
        key = sha256_text(seq)
        if int(key, 16) % shards == shard and key not in done:
            todo[key] = seq
    todo = sorted(todo.items(), key=lambda item: len(item[1]))
    start = time.time()
    for first in range(0, len(todo), part_size):
        part = todo[first : first + part_size]
        keys = [key for key, _ in part]
        vectors = embed([seq for _, seq in part])
        name = f"part_{sha256_text(''.join(keys))[:16]}.pt"
        atomic_write(
            out_dir / name, lambda tmp: torch.save({"keys": keys, "vectors": vectors}, tmp)
        )
        count = first + len(part)
        print(
            f"[esm_cache] {count}/{len(todo)} ({count / (time.time() - start):.1f} proteins/s)",
            flush=True,
        )
    return len(todo)


def load_cache(out_dir: str | Path):
    """Return (key -> row, fp32 table) of all parts."""
    import torch

    keys, vectors = [], []
    for part in sorted(Path(out_dir).glob("part_*.pt")):
        data = torch.load(part)
        keys += data["keys"]
        vectors.append(data["vectors"])
    return {key: i for i, key in enumerate(keys)}, torch.cat(vectors)


def check_cache(out_dir: str | Path, recipe: dict, sequences: list[str]) -> None:
    """Raise when the cache has another recipe or lacks one of the sequences."""
    manifest = Path(out_dir) / "manifest.json"
    if not manifest.exists() or read_json(manifest) != recipe:
        raise ValueError(
            f"No ESM cache with the current recipe in {out_dir}; run emulator_bench.esm_cache"
        )
    index, _ = load_cache(out_dir)
    missing = sum(sha256_text(s) not in index for s in set(sequences))
    if missing:
        raise ValueError(
            f"{missing} sequences are missing from {out_dir}; run emulator_bench.esm_cache"
        )


def install_frozen_esm(out_dir: str | Path) -> None:
    """Make FairEsm2 return the cached vector of each sequence instead of running ESM-2.

    The ESM-2 weights are loaded as before and then dropped: this keeps the random numbers of model creation, so seed,
    initial weights and batch draws equal the frozen original. Model state dicts then hold no ESM-2 weights.
    """
    import torch
    from clipzyme.models.fair_esm import FairEsm, FairEsm2

    original_init = FairEsm.__init__
    cache = {}

    def init(self, args):
        original_init(self, args)
        del self.model

    def forward(self, x):
        if not cache:
            cache["index"], cache["table"] = load_cache(out_dir)
        device = self.devicevar.device
        if cache["table"].device != device:
            cache["table"] = cache["table"].to(device)
        sequences = x["sequence"] if isinstance(x, dict) else x
        rows = torch.tensor([cache["index"][sha256_text(s)] for s in sequences], device=device)
        return {"hidden": cache["table"][rows]}

    FairEsm.__init__ = init
    FairEsm2.forward = forward


def collect_sequences(
    dataset_root: str | Path, max_length: int, groups: list[str] | None
) -> list[str]:
    """Unique sequences of at most max_length residues in the train, val and test rows of the splits."""
    from .dataset_adapter import SPLITS, discover_splits, read_rows, select_splits

    unique = set()
    for split in select_splits(discover_splits(dataset_root)[0], groups):
        for name in SPLITS:
            sequences = read_rows(split.path / f"{name}.parquet")[0]["sequence"]
            unique.update(sequences[sequences.str.len() <= max_length])
    return sorted(unique)


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> None:
    from .utils import load_hparams

    parser = argparse.ArgumentParser(
        description="Build the frozen ESM-2 vector cache for CLIPZyme"
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split_groups", nargs="*", help="Default is every split on disk")
    parser.add_argument("--hub_dir", default=None, help="Default <out_root>/weights/torch_hub")
    parser.add_argument(
        "--cache_dir", default=None, help="Default <out_root>/embeddings/esm_frozen_v1"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None, help="Smoke test only: first N sequences by length"
    )
    parser.add_argument("--no_sdpa", action="store_true", help="Run the original ESM-2 attention")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log", default=None, help="Default <out_root>/logs/esm_cache.log")
    args = parser.parse_args()
    log_path = Path(args.log or Path(args.out_root) / "logs" / "esm_cache.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = _Tee(sys.stdout, open(log_path, "a", buffering=1))
    hub_dir = args.hub_dir or Path(args.out_root) / "weights" / "torch_hub"
    sequences = collect_sequences(
        args.dataset_root, load_hparams()["max_protein_length"], args.split_groups
    )
    if args.limit:
        sequences = sorted(sequences, key=len)[: args.limit]
    print(f"[esm_cache] {len(sequences)} sequences, shard {args.shard}/{args.shards}", flush=True)
    model, alphabet = load_esm(hub_dir, args.device, not args.no_sdpa)
    build_cache(
        sequences,
        args.cache_dir or cache_dir(args.out_root),
        lambda seqs: pool_sequences(model, alphabet, seqs, args.device),
        fingerprint(hub_dir),
        args.shard,
        args.shards,
    )


if __name__ == "__main__":
    main()
