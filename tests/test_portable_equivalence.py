"""Equivalence tests that need no downloaded weights and no private data.

ESM-2 is a small random model of the same fair-esm classes. The reactions and proteins come from
tests/fixtures/tiny_enzyme_reactions.json. Needs fair-esm 2.0.0 (env clipzyme).
"""

import copy
import json
import os
import types
from pathlib import Path

import pytest
import torch

from emulator_bench import esm_cache, fast_esm, virtual_ranks

esm = pytest.importorskip("esm")
from clipzyme.learning.losses.contrastive import CLIPLoss  # noqa: E402
from clipzyme.models.fair_esm import FairEsm, FairEsm2  # noqa: E402
from clipzyme.models.protmol import EnzymeReactionCLIP  # noqa: E402
from clipzyme.utils.loading import default_collate  # noqa: E402
from clipzyme.utils.pyg import from_mapped_smiles  # noqa: E402
from tests.test_frozen_esm import LOSS_ARGS, clip_args  # noqa: E402
from tests.test_fast_esm import tiny_encoder  # noqa: E402

ROWS = json.loads((Path(__file__).parent / "fixtures" / "tiny_enzyme_reactions.json").read_text())
SEQS = [r["sequence"] for r in ROWS]
LAYERS = 3
K = 10  # the production gap must stay within 10x the production-path noise (perturbed weights) gap
NOISE_U = 2**-8  # one rounding of the weights at the production precision (bf16 unit roundoff)
NEGATIVE_CONTROL = os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1"


@pytest.fixture(autouse=True)
def restore_patches(monkeypatch):
    import clipzyme.models.chemprop as chemprop
    import esm.modules
    from esm.multihead_attention import MultiheadAttention

    for owner, name in [
        (CLIPLoss, "__call__"),
        (EnzymeReactionCLIP, "encode_reaction"),
        (FairEsm, "__init__"),
        (FairEsm2, "forward"),
        (MultiheadAttention, "forward"),
        (esm.modules, "gelu"),
        (esm.modules.TransformerLayer, "forward"),
        (chemprop, "get_reverse_edge_indices"),
    ]:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(fast_esm, "recompute", False)


@pytest.mark.parametrize("token_dropout", [False, True])
def test_cache_vectors_equal_in_loop_esm_on_long_and_short_proteins(token_dropout):
    """Portable form of the real-weights test: proteins of 3 to 150 residues, padded batch vs sorted chunks with SDPA.

    Real ESM-2 has token_dropout=True, which rescales the embeddings by the share of unmasked tokens per sequence.
    """
    encoder = tiny_encoder()
    encoder.model.token_dropout = token_dropout
    with torch.no_grad():
        original = FairEsm.forward(encoder, SEQS)["hidden"]
    fast_esm.patch_esm_kernels()
    pooled = esm_cache.pool_sequences(
        encoder.model, encoder.alphabet, SEQS, "cpu", layer=LAYERS, budget=300
    )
    assert (original - pooled).abs().max().item() < 1e-5


def make_batches(batch_size):
    items = [
        {
            "reactants": from_mapped_smiles(r["reactants"], encode_no_edge=True)[0],
            "products": from_mapped_smiles(r["products"], encode_no_edge=True)[0],
            "sequence": r["sequence"],
        }
        for r in ROWS
    ]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def train(tiny, alphabet, steps, device, autocast, cache=None, noise=False, scale=1.0):
    with pytest.MonkeyPatch.context() as mp:  # restores the cache patches, so cached runs can repeat
        if cache is not None:
            mp.setattr(FairEsm, "__init__", FairEsm.__init__)
            mp.setattr(FairEsm2, "forward", FairEsm2.forward)
            esm_cache.install_frozen_esm(cache)
        return train_once(tiny, alphabet, steps, device, autocast, noise, scale)


def train_once(tiny, alphabet, steps, device, autocast, noise, scale):
    """Train a frozen-ESM CLIP model for `steps` AdamW steps on the first batch with 4 virtual ranks.

    Returns the per-step losses and the loss of the second batch after training.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.hub, "load", lambda *_: (copy.deepcopy(tiny), alphabet))
        mp.setattr(torch.hub, "set_dir", lambda *_: None)
        torch.manual_seed(1)
        model = EnzymeReactionCLIP(clip_args()).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if noise:
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in trainable:
                p += NOISE_U * p.abs() * torch.randn(p.shape, generator=generator).to(p.device)
    if scale != 1.0:
        with torch.no_grad():
            for p in trainable:
                p *= scale
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    train_items, eval_items = make_batches(4)[0] * 2, make_batches(4)[1] * 2

    def loss_of(items):
        batch = default_collate(items)
        batch["reactants"], batch["products"] = batch["reactants"].to(device), batch["products"].to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            output = model(batch)
            loss, _, _ = CLIPLoss()(output, batch, types.SimpleNamespace(model=model), LOSS_ARGS)
        return loss

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_of(train_items)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        return losses, loss_of(eval_items).item()


def reference_and_candidate(tmp_path, steps, device, autocast, noise=False, scale=1.0, token_dropout=False):
    torch.manual_seed(0)
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    tiny = esm.model.esm2.ESM2(
        num_layers=LAYERS, embed_dim=8, attention_heads=2, alphabet=alphabet, token_dropout=token_dropout
    )
    virtual_ranks.install(4)
    reference = train(tiny, alphabet, steps, device, autocast)
    pooled = esm_cache.pool_sequences(tiny.cpu().float(), alphabet, SEQS, "cpu", layer=LAYERS)
    esm_cache.build_cache(SEQS, tmp_path, lambda s: pooled[[SEQS.index(x) for x in s]], {})
    fast_esm.patch_esm_kernels()
    fast_esm.patch_reverse_edges()
    candidate = train(tiny, alphabet, steps, device, autocast, cache=tmp_path, scale=scale)
    clean = train(tiny, alphabet, steps, device, autocast, cache=tmp_path) if scale != 1.0 else candidate
    noisy = train(tiny, alphabet, steps, device, autocast, cache=tmp_path, noise=True) if noise else None
    return reference, candidate, noisy, clean


@pytest.mark.parametrize("token_dropout", [False, True])
def test_cached_training_follows_in_loop_esm_training_fp32(tmp_path, token_dropout):
    """Frozen cache path vs in-loop frozen ESM, fp32, 4 virtual ranks, 5 AdamW steps.

    Step 0 differs only by fp32 rounding of the ESM forward (bound 1e-5). AdamW amplifies that rounding in later
    steps; the measured growth on the real run was 1e-3 over 42 steps, the same as a 1e-6 noise control. The bound
    for 5 steps is 1e-4 for the training losses and for the loss on the held-out batch.
    """
    (ref_losses, ref_eval), (cand_losses, cand_eval), _, _ = reference_and_candidate(
        tmp_path, 5, "cpu", autocast=False, token_dropout=token_dropout
    )
    assert abs(ref_losses[0] - cand_losses[0]) / abs(ref_losses[0]) < 1e-5
    for ref, cand in zip(ref_losses, cand_losses):
        assert abs(ref - cand) / abs(ref) < 1e-4
    assert abs(ref_eval - cand_eval) / abs(ref_eval) < 1e-4


@pytest.mark.gpu
def test_production_path_within_noise_control(tmp_path):
    """Production (--frozen_esm --amp --fast: fp32 ESM cache, bf16 autocast) vs original (in-loop ESM, bf16 autocast).

    clip_esm_frozen.json trains in bf16, so the original precision is bf16 too. The two paths differ in the ESM
    vector (fp32 cache vs bf16 in-loop forward) and in kernels. A noise control reruns the production path with the
    initial weights perturbed by 2**-8 (relative, one bf16 rounding). Floor = |perturbed production - unscaled production|,
    gap = |production - original|. The per-step loss gap (20 steps, 4 virtual ranks) and the held-out loss gap must stay
    within K times the floor. Step 0 is judged like every other step.
    The ESM-2 has token_dropout=True, as the real model. With EQUIV_NEGATIVE_CONTROL=1 the production initial weights
    are multiplied by 1.05 and the test must fail.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    (ref, ref_eval), (cand, cand_eval), (noise, noise_eval), (clean, clean_eval) = reference_and_candidate(
        tmp_path, 20, "cuda", autocast=True, noise=True, scale=1.05 if NEGATIVE_CONTROL else 1.0, token_dropout=True
    )
    ref, cand, noise, clean = (torch.tensor(x, dtype=torch.float64) for x in (ref, cand, noise, clean))
    tiny = 1e-7 * ref.abs().max()
    gap, floor = (cand - ref).abs(), (noise - clean).abs()
    final_gap, final_floor = abs(cand_eval - ref_eval), abs(noise_eval - clean_eval)
    tiny0 = 1e-7 * ref[0].abs()
    gap0, floor0 = float(gap[0]), float(floor[0])
    print(
        f"clipzyme bf16: max gap={gap.max():.3e}, max floor={floor.max():.3e}, "
        f"ratio={gap.max() / max(floor.max(), tiny):.2f}, K={K}, step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}, "
        f"final gap={final_gap:.3e}, final floor={final_floor:.3e}, "
        f"final ratio={final_gap / max(final_floor, 1e-7 * abs(ref_eval)):.2f}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap.max() <= K * max(floor.max(), tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * abs(ref_eval))
