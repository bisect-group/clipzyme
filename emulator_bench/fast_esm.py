"""Exact speed-ups for CLIPZyme training and evaluation, applied at run time (the upstream package stays unchanged).

- ESM-2 attention uses scaled_dot_product_attention and the MLP uses F.gelu (same formulas, fused kernels).
- Inside one micro-batch, the ESM forward runs in length-sorted chunks of about CHUNK_TOKENS tokens.
  The micro-batch, the batch order and the contrastive negatives do not change. Outputs return in the original order.
- ESM-2 layers recompute activations in the backward pass only when a micro-batch has more than CHECKPOINT_TOKENS tokens.
- get_reverse_edge_indices is vectorised (sort and searchsorted).
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK_TOKENS = 1024
CHECKPOINT_TOKENS = 4000

# Whether ESM layers recompute activations now. The ESM forward sets it for each micro-batch.
recompute = False


def sdpa_attention_forward(self, query, key, value, key_padding_mask=None, **unused):
    """Replacement for esm MultiheadAttention.forward (self-attention with rotary embeddings, no attention weights)."""
    T, B, E = query.shape
    H, D = self.num_heads, self.head_dim
    q, k, v = (
        proj(query).contiguous().view(T, B * H, D).transpose(0, 1)
        for proj in (self.q_proj, self.k_proj, self.v_proj)
    )
    q, k = self.rot_emb(q, k)  # sdpa scales q by D**-0.5, equal to self.scaling for D = 64
    q, k, v = (t.view(B, H, T, D) for t in (q, k, v))
    mask = None if key_padding_mask is None else ~key_padding_mask.to(torch.bool)[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return self.out_proj(out.permute(2, 0, 1, 3).reshape(T, B, E)), None


def patch_esm_kernels() -> None:
    import esm.modules
    from esm.multihead_attention import MultiheadAttention

    MultiheadAttention.forward = sdpa_attention_forward
    esm.modules.gelu = F.gelu


def token_chunks(lengths: list[int], budget: int = CHUNK_TOKENS) -> list[list[int]]:
    """Indices sorted by length, grouped so that count * (longest + 2 special tokens) stays within budget."""
    chunks: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if chunks and (len(chunks[-1]) + 1) * (lengths[i] + 2) <= budget:
            chunks[-1].append(i)
        else:
            chunks.append([i])
    return chunks


def chunked_forward(self, x):
    """FairEsm.forward that returns {"hidden": mean ESM representation per sequence} in input order."""
    global recompute
    if self.args.freeze_esm or self.use_cls_token or self.args.esm_return_contacts:
        raise ValueError(
            "--fast does not support freeze_esm, use_esm_cls_token or esm_return_contacts"
        )
    if isinstance(x, dict):
        x = x["sequence"]
    pairs = self.truncate_protein(x)
    x = [seq for _, seq in pairs]
    chunks = token_chunks([len(s) for s in x], CHUNK_TOKENS)
    recompute = self.always_recompute or sum(map(len, x)) + 2 * len(x) > CHECKPOINT_TOKENS
    hidden = []
    for chunk in chunks:
        _, _, tokens = self.batch_converter([(i, x[i]) for i in chunk])
        tokens = tokens.to(self.devicevar.device)
        out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        mask = (
            (tokens != self.alphabet.cls_idx)
            & (tokens != self.alphabet.eos_idx)
            & (tokens != self.alphabet.padding_idx)
        )
        mask = mask.long().unsqueeze(-1)
        hidden.append((out["representations"][self.repr_layer] * mask).sum(1) / mask.sum(1))
    hidden = torch.cat(hidden)
    order = torch.tensor([i for chunk in chunks for i in chunk], device=hidden.device)
    restored = torch.empty_like(hidden)
    restored[order] = hidden
    return {"hidden": restored}


def patch_layer_recompute() -> None:
    from esm.modules import TransformerLayer

    layer_forward = TransformerLayer.forward

    def forward(self, x, *args, **kwargs):
        if not (recompute and self.training and torch.is_grad_enabled()):
            return layer_forward(self, x, *args, **kwargs)
        return checkpoint(layer_forward, self, x, *args, use_reentrant=False, **kwargs)

    TransformerLayer.forward = forward


def reverse_edge_indices(edge_index: torch.Tensor) -> torch.Tensor:
    """For each edge i->j, the index of the edge j->i."""
    src, dst = edge_index
    base = edge_index.max() + 1
    keys, order = (src * base + dst).sort()
    return order[torch.searchsorted(keys, dst * base + src)]


def patch_reverse_edges() -> None:
    import clipzyme.models.chemprop as chemprop

    chemprop.get_reverse_edge_indices = reverse_edge_indices


def install_training(always_recompute: bool) -> None:
    """Apply every training speed-up. always_recompute keeps --checkpoint_esm semantics (recompute for every micro-batch)."""
    from clipzyme.models.fair_esm import FairEsm2

    patch_esm_kernels()
    patch_layer_recompute()
    patch_reverse_edges()
    FairEsm2.always_recompute = always_recompute
    FairEsm2.forward = chunked_forward
