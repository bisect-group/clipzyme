"""One-GPU emulation of the multi-GPU CLIP loss of clipzyme/learning/losses/contrastive.py.

With clip_loss_use_gather and several GPUs, each GPU contrasts its local rows against the features of all GPUs.
concat_all_gather has no gradient, so the gathered features are constants and only the local rows carry gradient.
DDP then averages the gradients of the GPUs. Here one forward covers the samples of all GPUs. The batch splits into
virtual ranks and the loss is the mean of the per-rank losses, which gives the same gradient as the averaged DDP gradient.
The reaction encoder runs on each virtual rank separately: one padded batch of all samples would need a dense
[graphs, atoms, atoms, 1280] tensor as large as the biggest graph of the whole step, and the model has no batch-dependent layer.
"""

import torch
import torch.nn.functional as F


def rank_rows(count: int, ranks: int, device=None) -> tuple[torch.Tensor, ...]:
    return torch.arange(count, device=device).tensor_split(ranks)


def install(ranks: int) -> None:
    from clipzyme.learning.losses.contrastive import CLIPLoss
    from clipzyme.models.protmol import EnzymeReactionCLIP
    from torch_geometric.data import Batch

    encode_reaction = EnzymeReactionCLIP.encode_reaction

    def encode_by_rank(self, batch):
        reactants, products = batch["reactants"], batch["products"]
        parts = []
        for rows in rank_rows(reactants.num_graphs, ranks):
            rows = rows.tolist()
            sub = {
                "reactants": Batch.from_data_list(reactants.index_select(rows)),
                "products": Batch.from_data_list(products.index_select(rows)),
            }
            parts.append(encode_reaction(self, sub))
        return torch.cat(parts)

    def call(self, model_output, batch, model, args):
        substrate = model_output["substrate_hiddens"]
        protein = model_output["protein_hiddens"]
        scale = model.model.logit_scale.exp()
        logits_per_substrate = (scale * substrate) @ protein.detach().t()
        logits_per_protein = (scale * protein) @ substrate.detach().t()
        losses = []
        for rows in rank_rows(len(substrate), ranks, substrate.device):
            smoothing = args.clip_loss_label_smoothing
            losses.append(
                (
                    F.cross_entropy(logits_per_protein[rows], rows, label_smoothing=smoothing)
                    + F.cross_entropy(logits_per_substrate[rows], rows, label_smoothing=smoothing)
                )
                / 2
            )
        loss = torch.stack(losses).mean()
        labels = torch.arange(len(substrate), device=substrate.device)
        probs = F.softmax(logits_per_substrate, dim=-1).detach()
        predictions = {
            "clip_probs": probs,
            "clip_preds": probs.argmax(axis=-1).reshape(-1),
            "clip_golds": labels,
        }
        return loss, {"clip_loss": loss.detach()}, predictions

    CLIPLoss.__call__ = call
    EnzymeReactionCLIP.encode_reaction = encode_by_rank
