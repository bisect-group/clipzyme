"""ESM cache, frozen-ESM hook and virtual-GPU loss."""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from emulator_bench import esm_cache, virtual_ranks
from emulator_bench.utils import sha256_text

esm = pytest.importorskip("esm")
from clipzyme.learning.losses.contrastive import CLIPLoss  # noqa: E402
from clipzyme.models.fair_esm import FairEsm, FairEsm2  # noqa: E402
from clipzyme.models.protmol import EnzymeReactionCLIP  # noqa: E402
from tests.test_fast_esm import SEQS, tiny_encoder  # noqa: E402


@pytest.fixture(autouse=True)
def restore_patches(monkeypatch):
    monkeypatch.setattr(CLIPLoss, "__call__", CLIPLoss.__call__)
    monkeypatch.setattr(EnzymeReactionCLIP, "encode_reaction", EnzymeReactionCLIP.encode_reaction)
    monkeypatch.setattr(FairEsm, "__init__", FairEsm.__init__)
    monkeypatch.setattr(FairEsm2, "forward", FairEsm2.forward)


def fake_embed(calls):
    def embed(seqs):
        calls.append(list(seqs))
        return torch.tensor([[float(len(s)), float(sum(map(ord, s)))] for s in seqs])

    return embed


def test_cache_resumes_shards_and_refuses_other_recipe(tmp_path):
    calls = []
    seqs = ["MK", "MKV", "MKVL", "MKVLA", "MKVLAA"]
    recipe = {"model": "a"}
    esm_cache.build_cache(
        seqs, tmp_path, fake_embed(calls), recipe, shard=0, shards=2, part_size=2
    )
    esm_cache.build_cache(
        seqs, tmp_path, fake_embed(calls), recipe, shard=1, shards=2, part_size=2
    )
    index, table = esm_cache.load_cache(tmp_path)
    assert sorted(index) == sorted(sha256_text(s) for s in seqs)
    assert sum(len(c) for c in calls) == len(seqs)  # shards do not overlap
    assert table[index[sha256_text("MKV")]].tolist() == [3.0, float(sum(map(ord, "MKV")))]

    calls.clear()
    assert esm_cache.build_cache(seqs + ["MKVLAAA"], tmp_path, fake_embed(calls), recipe) == 1
    assert calls == [["MKVLAAA"]]  # finished parts are not computed again
    esm_cache.check_cache(tmp_path, recipe, seqs)
    with pytest.raises(ValueError, match="missing"):
        esm_cache.check_cache(tmp_path, recipe, seqs + ["NEW"])
    with pytest.raises(ValueError, match="recipe"):
        esm_cache.check_cache(tmp_path, {"model": "b"}, seqs)
    with pytest.raises(ValueError, match="recipe"):
        esm_cache.build_cache(seqs, tmp_path, fake_embed(calls), {"model": "b"})


def test_cached_vector_equals_original_fair_esm_output():
    encoder = tiny_encoder()
    seqs = SEQS + ["MKTAYIAKQRQISFVKSHFSRQMKTAYIAKQRQISFVK"]
    with torch.no_grad():
        original = FairEsm.forward(encoder, seqs)["hidden"]
    pooled = esm_cache.pool_sequences(
        encoder.model, encoder.alphabet, seqs, "cpu", layer=3, budget=60
    )
    assert torch.allclose(original, pooled, rtol=1e-5, atol=1e-6)


def clip_args(**extra):
    return types.SimpleNamespace(
        reaction_clip_model_path=None,
        model_name="enzyme_reaction_clip",
        protein_encoder="fair_esm2",
        use_protein_graphs=False,
        do_matching_task=False,
        clip_freeze_esm=True,
        pretrained_hub_dir="unused",
        esm_name="tiny",
        freeze_esm=False,
        esm_hidden_layer=3,
        use_esm_cls_token=False,
        esm_return_contacts=False,
        chemprop_hidden_dim=8,
        chemprop_node_dim=9,
        chemprop_edge_dim=3,
        chemprop_num_layers=2,
        chemprop_pool="none",
        aggregate_over_edges=False,
        **extra,
    )


def test_frozen_hook_keeps_random_numbers_and_returns_original_vectors(tmp_path, monkeypatch):
    torch.manual_seed(0)
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    tiny = esm.model.esm2.ESM2(
        num_layers=3, embed_dim=8, attention_heads=2, alphabet=alphabet, token_dropout=False
    )

    def tiny_hub(*unused):
        torch.rand(100)  # model creation uses random numbers
        return tiny, alphabet

    monkeypatch.setattr(torch.hub, "load", tiny_hub)
    monkeypatch.setattr(torch.hub, "set_dir", lambda *_: None)

    torch.manual_seed(1)
    original = EnzymeReactionCLIP(clip_args())
    rng_after_original = torch.get_rng_state()
    batch = {"sequence": SEQS}
    with torch.no_grad():
        expected = original.encode_protein(batch)
    pooled = esm_cache.pool_sequences(tiny, alphabet, SEQS, "cpu", layer=3)
    esm_cache.build_cache(SEQS, tmp_path, lambda seqs: pooled[[SEQS.index(s) for s in seqs]], {})

    esm_cache.install_frozen_esm(tmp_path)
    torch.manual_seed(1)
    frozen = EnzymeReactionCLIP(clip_args())
    assert torch.equal(torch.get_rng_state(), rng_after_original)
    assert not any(k.startswith("protein_encoder.model") for k in frozen.state_dict())
    trainable = [k for k, p in original.named_parameters() if p.requires_grad]
    assert trainable == [k for k, p in frozen.named_parameters() if p.requires_grad]
    for key in trainable:
        assert torch.equal(original.state_dict()[key], frozen.state_dict()[key])
    with torch.no_grad():
        assert torch.allclose(frozen.encode_protein(batch), expected, rtol=1e-5, atol=1e-6)


RANKS, LOCAL, DIM = 4, 8, 6


def make_features():
    torch.manual_seed(0)
    return torch.randn(RANKS * LOCAL, DIM), torch.randn(RANKS * LOCAL, DIM)


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.model = torch.nn.Module()
        self.model.logit_scale = torch.nn.Parameter(torch.tensor(2.5))
        self.model.s = torch.nn.Linear(DIM, DIM)
        self.model.p = torch.nn.Linear(DIM, DIM)

    def outputs(self, substrate, protein):
        norm = torch.nn.functional.normalize
        return {
            "substrate_hiddens": norm(self.model.s(substrate), dim=1),
            "protein_hiddens": norm(self.model.p(protein), dim=1),
        }


LOSS_ARGS = types.SimpleNamespace(
    clip_loss_use_gather=True, gpus=RANKS, clip_loss_label_smoothing=0.0
)


def ddp_rank(rank, path, queue):
    """The original CLIPLoss on one of four gloo processes; DDP averages the gradients."""
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=RANKS)
    substrate, protein = make_features()
    tower = Tower()
    rows = slice(rank * LOCAL, (rank + 1) * LOCAL)
    output = tower.outputs(substrate[rows], protein[rows])
    loss, _, _ = CLIPLoss()(output, None, tower, LOSS_ARGS)
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in tower.parameters()])
    dist.all_reduce(grads)
    queue.put((rank, loss.item(), (grads / RANKS).tolist()))


def test_virtual_ranks_match_four_process_ddp_gradients(monkeypatch):
    path = tempfile.mktemp()
    queue = mp.get_context("spawn").SimpleQueue()
    context = mp.get_context("spawn")
    procs = [context.Process(target=ddp_rank, args=(r, path, queue)) for r in range(RANKS)]
    for p in procs:
        p.start()
    results = {r: (loss, grads) for r, loss, grads in (queue.get() for _ in procs)}
    for p in procs:
        p.join()
        assert p.exitcode == 0
    os.path.exists(path) and os.remove(path)

    virtual_ranks.install(RANKS)
    substrate, protein = make_features()
    tower = Tower()
    loss, _, predictions = CLIPLoss()(tower.outputs(substrate, protein), None, tower, LOSS_ARGS)
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in tower.parameters()])
    assert loss.item() == pytest.approx(
        sum(loss for loss, _ in results.values()) / RANKS, rel=1e-6
    )
    assert torch.allclose(grads, torch.tensor(results[0][1]), rtol=1e-5, atol=1e-7)
    assert predictions["clip_probs"].shape == (RANKS * LOCAL, RANKS * LOCAL)


REACTIONS = [
    (
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH3:7]",
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH3:7].[OH2:4]",
    ),
    ("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH:2]=[O:3]"),
    ("[CH3:1][CH2:2][CH2:3][OH:4]", "[CH3:1][CH2:2][CH:3]=[O:4]"),
]


def test_reaction_encoder_by_rank_equals_one_batch(monkeypatch):
    from clipzyme.utils.loading import default_collate
    from clipzyme.utils.pyg import from_mapped_smiles

    torch.manual_seed(0)
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    tiny = esm.model.esm2.ESM2(
        num_layers=1, embed_dim=8, attention_heads=2, alphabet=alphabet, token_dropout=False
    )
    monkeypatch.setattr(torch.hub, "load", lambda *_: (tiny, alphabet))
    monkeypatch.setattr(torch.hub, "set_dir", lambda *_: None)
    model = EnzymeReactionCLIP(clip_args()).eval()
    items = [
        {
            "reactants": from_mapped_smiles(r, encode_no_edge=True)[0],
            "products": from_mapped_smiles(p, encode_no_edge=True)[0],
        }
        for r, p in REACTIONS * 3
    ]
    batch = default_collate(items)
    with torch.no_grad():
        whole = model.encode_reaction(batch)
        virtual_ranks.install(4)
        by_rank = model.encode_reaction(default_collate(items))
    assert whole.shape == by_rank.shape == (9, 8)
    assert torch.allclose(whole, by_rank, rtol=1e-5, atol=1e-6)


UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
LYSOZYME = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALL"
    "SSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFF"
    "KSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRH"
    "NIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)
HUB_DIR = esm_cache.DEFAULT_OUT_ROOT / "weights" / "torch_hub"


def test_cache_vectors_equal_in_loop_esm2_on_real_sequences(monkeypatch):
    """Real ESM-2 650M, different lengths, padded batch vs length-sorted chunks with SDPA."""
    import esm.modules
    from esm.multihead_attention import MultiheadAttention

    if not (HUB_DIR / "checkpoints").exists():
        pytest.skip("ESM-2 weights are not downloaded")
    monkeypatch.setattr(MultiheadAttention, "forward", MultiheadAttention.forward)
    monkeypatch.setattr(esm.modules, "gelu", esm.modules.gelu)
    monkeypatch.setattr(
        esm.modules.TransformerLayer, "forward", esm.modules.TransformerLayer.forward
    )
    torch.hub.set_dir(str(HUB_DIR))
    model, alphabet = torch.hub.load("facebookresearch/esm:v2.0.0", esm_cache.ESM_NAME)
    model.eval()
    encoder = types.SimpleNamespace(
        model=model,
        alphabet=alphabet,
        batch_converter=alphabet.get_batch_converter(),
        devicevar=torch.zeros(1),
        repr_layer=33,
        args=types.SimpleNamespace(freeze_esm=True, esm_return_contacts=False),
        use_cls_token=False,
        truncate_protein=lambda x: FairEsm.truncate_protein(None, x),
    )
    seqs = [GFP, UBIQUITIN, LYSOZYME]
    with torch.no_grad():
        original = FairEsm.forward(encoder, seqs)["hidden"]  # one padded batch
    from emulator_bench.fast_esm import patch_esm_kernels

    patch_esm_kernels()
    pooled = esm_cache.pool_sequences(model, alphabet, seqs, "cpu", budget=300)
    assert (original - pooled).abs().max().item() < 1e-5


def test_step_zero_loss_and_gradients_equal_in_loop_esm(monkeypatch):
    from clipzyme.utils.loading import default_collate
    from clipzyme.utils.pyg import from_mapped_smiles

    torch.manual_seed(0)
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    tiny = esm.model.esm2.ESM2(
        num_layers=3, embed_dim=8, attention_heads=2, alphabet=alphabet, token_dropout=False
    )
    monkeypatch.setattr(torch.hub, "load", lambda *_: (tiny, alphabet))
    monkeypatch.setattr(torch.hub, "set_dir", lambda *_: None)
    seqs = SEQS + [SEQS[0]]
    items = [
        {
            "reactants": from_mapped_smiles(r, encode_no_edge=True)[0],
            "products": from_mapped_smiles(p, encode_no_edge=True)[0],
            "sequence": s,
        }
        for (r, p), s in zip(REACTIONS * 2, seqs)
    ]
    virtual_ranks.install(4)

    def step(model):
        batch = default_collate(items)
        model.zero_grad()
        output = model(batch)
        loss, _, _ = CLIPLoss()(output, batch, types.SimpleNamespace(model=model), LOSS_ARGS)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return loss.item(), grads

    torch.manual_seed(1)
    ref_loss, ref_grads = step(EnzymeReactionCLIP(clip_args()))
    with tempfile.TemporaryDirectory() as cache:
        pooled = esm_cache.pool_sequences(tiny, alphabet, seqs, "cpu", layer=3)
        esm_cache.build_cache(seqs, cache, lambda s: pooled[[seqs.index(x) for x in s]], {})
        esm_cache.install_frozen_esm(cache)
        torch.manual_seed(1)
        cand_loss, cand_grads = step(EnzymeReactionCLIP(clip_args()))
    assert abs(ref_loss - cand_loss) / abs(ref_loss) < 1e-5
    assert ref_grads.keys() == cand_grads.keys()
    for name, grad in ref_grads.items():
        assert (grad - cand_grads[name]).abs().max().item() < 1e-5, name
