import types

import pytest
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from emulator_bench import fast_esm

esm = pytest.importorskip("esm")
from clipzyme.models.fair_esm import FairEsm  # noqa: E402
from esm.modules import TransformerLayer  # noqa: E402

SEQS = ["MKTAYIAKQR", "MKV", "MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIA", "MSKGEELFTG"]


@pytest.fixture(autouse=True)
def restore_patches(monkeypatch):
    import esm.modules
    from esm.multihead_attention import MultiheadAttention

    monkeypatch.setattr(MultiheadAttention, "forward", MultiheadAttention.forward)
    monkeypatch.setattr(esm.modules, "gelu", esm.modules.gelu)
    monkeypatch.setattr(TransformerLayer, "forward", TransformerLayer.forward)
    monkeypatch.setattr(fast_esm, "recompute", False)


def tiny_encoder() -> types.SimpleNamespace:
    """Stands in for a FairEsm2 instance with a small ESM-2 model."""
    torch.manual_seed(0)
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    model = esm.model.esm2.ESM2(
        num_layers=3, embed_dim=64, attention_heads=2, alphabet=alphabet, token_dropout=False
    ).train()
    return types.SimpleNamespace(
        model=model,
        alphabet=alphabet,
        batch_converter=alphabet.get_batch_converter(),
        devicevar=torch.zeros(1),
        repr_layer=3,
        args=types.SimpleNamespace(freeze_esm=False, esm_return_contacts=False),
        use_cls_token=False,
        always_recompute=False,
        truncate_protein=lambda x: FairEsm.truncate_protein(None, x),
    )


def run(forward, encoder, seqs=SEQS):
    encoder.model.zero_grad()
    hidden = forward(encoder, seqs)["hidden"]
    weights = torch.linspace(-1, 1, hidden.numel()).view_as(hidden)
    (hidden * weights).sum().backward()
    grads = torch.cat([p.grad.flatten() for p in encoder.model.parameters() if p.grad is not None])
    return hidden.detach().clone(), grads.clone()


def assert_close(a, b, rel=1e-5):
    assert ((a - b).abs().max() / a.abs().max()).item() < rel


def test_sdpa_and_gelu_match_original_outputs_and_grads():
    encoder = tiny_encoder()
    hidden, grads = run(FairEsm.forward, encoder)
    fast_esm.patch_esm_kernels()
    fast_hidden, fast_grads = run(FairEsm.forward, encoder)
    assert_close(hidden, fast_hidden)
    assert_close(grads, fast_grads)


def test_chunked_forward_returns_original_order(monkeypatch):
    encoder = tiny_encoder()
    hidden, grads = run(FairEsm.forward, encoder)
    monkeypatch.setattr(fast_esm, "CHUNK_TOKENS", 40)
    assert len(fast_esm.token_chunks([len(s) for s in SEQS], 40)) > 1
    fast_hidden, fast_grads = run(fast_esm.chunked_forward, encoder)
    assert_close(hidden, fast_hidden)
    assert_close(grads, fast_grads)


def test_token_chunks_sort_by_length_within_budget():
    chunks = fast_esm.token_chunks([5, 1, 9, 2], budget=24)
    assert chunks == [[1, 3, 0], [2]]


def test_adaptive_recompute_matches_plain_grads(monkeypatch):
    encoder = tiny_encoder()
    fast_esm.patch_layer_recompute()
    calls = []
    real_checkpoint = fast_esm.checkpoint
    monkeypatch.setattr(
        fast_esm, "checkpoint", lambda *a, **k: calls.append(1) or real_checkpoint(*a, **k)
    )
    hidden, grads = run(fast_esm.chunked_forward, encoder)
    assert not calls  # the micro-batch is below CHECKPOINT_TOKENS
    monkeypatch.setattr(fast_esm, "CHECKPOINT_TOKENS", 10)
    hidden_ck, grads_ck = run(fast_esm.chunked_forward, encoder)
    assert len(calls) == 3  # one per layer
    assert_close(hidden, hidden_ck, 1e-6)
    assert_close(grads, grads_ck, 1e-6)


def original_reverse_edges(edge_index):
    from clipzyme.models.chemprop import get_reverse_edge_indices

    return get_reverse_edge_indices(edge_index)


def molecule_edges(smiles_list):
    from rdkit import Chem

    edges, offset = [], 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx() + offset, bond.GetEndAtomIdx() + offset
            edges += [(i, j), (j, i)]
        offset += mol.GetNumAtoms()
    return torch.tensor(edges).T, offset


def test_reverse_edge_indices_match_original_on_reaction_graphs():
    reactant, n = molecule_edges(["CCO", "c1ccccc1C"])
    product, _ = molecule_edges(["C1CO1", "C1CCCCC1C"])
    adjacency = lambda e: to_dense_adj(e, max_num_nodes=n)[0]  # noqa: E731
    difference = dense_to_sparse(adjacency(reactant) + adjacency(product))[0]
    shuffled = reactant[
        :, torch.randperm(reactant.shape[1], generator=torch.Generator().manual_seed(0))
    ]
    for edge_index in (reactant, product, difference, shuffled):
        assert torch.equal(
            original_reverse_edges(edge_index), fast_esm.reverse_edge_indices(edge_index)
        )


def test_encode_by_length_restores_input_order():
    from emulator_bench.evaluate import encode

    items = ["AAAA", "A", "AAA", "AA", "AAAAA"]
    embed = lambda batch: torch.tensor([[float(len(s))] for s in batch["sequence"]])  # noqa: E731
    collate = lambda s: {"sequence": s}  # noqa: E731
    import clipzyme.utils.loading as loading

    batches = []

    def spy(samples):
        batches.append([s["sequence"] for s in samples])
        return {"sequence": [s["sequence"] for s in samples]}

    original = loading.default_collate
    loading.default_collate = spy
    try:
        out = encode(items, collate, embed, 2, "cpu", False, by_length=True)
    finally:
        loading.default_collate = original
    assert batches == [["A", "AA"], ["AAA", "AAAA"], ["AAAAA"]]
    assert out.flatten().tolist() == [4.0, 1.0, 3.0, 2.0, 5.0]


def test_long_sequence_is_truncated_like_the_original():
    encoder = tiny_encoder()
    seqs = ["MKTAYIAKQR" * 110, "MKV"]
    hidden, _ = run(FairEsm.forward, encoder, seqs)
    fast_hidden, _ = run(fast_esm.chunked_forward, encoder, seqs)
    assert_close(hidden, fast_hidden)


def test_list_wrapped_items_work():
    encoder = tiny_encoder()
    seqs = [[s] for s in SEQS]
    hidden, _ = run(FairEsm.forward, encoder, seqs)
    fast_hidden, _ = run(fast_esm.chunked_forward, encoder, seqs)
    assert_close(hidden, fast_hidden)


@pytest.mark.parametrize("flag", ["freeze_esm", "esm_return_contacts", "use_cls_token"])
def test_unsupported_flags_raise(flag):
    encoder = tiny_encoder()
    if flag == "use_cls_token":
        encoder.use_cls_token = True
    else:
        setattr(encoder.args, flag, True)
    with pytest.raises(ValueError, match="--fast does not support"):
        fast_esm.chunked_forward(encoder, SEQS)
