from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


def pool_sum(values, owners, valid, sentences):
    result = values.new_zeros((sentences, *values.shape[1:]))
    return result.index_add(0, owners[valid], values[valid])


def balanced_sentence_ce(logits, indices, labels):
    terms = []
    for label in sorted(set(labels)):
        chosen = [index for (index, target) in zip(indices, labels) if target == label]
        target = torch.full(
            (len(chosen),), label, dtype=torch.long, device=logits.device
        )
        terms.append(F.cross_entropy(logits[chosen].float(), target))
    return torch.stack(terms).mean()


class SentenceEvidenceModel(nn.Module):
    """One query per forward; all original body tokens contribute exactly once."""

    def __init__(self, official, variant, special_ids, width=256):
        super().__init__()
        if variant not in {"adapted_provence", "without_pe"}:
            raise ValueError(variant)
        self.variant = variant
        self.use_sentence_interaction = variant == "without_pe"
        self.use_sentence_position_encoding = False
        self.last_losses = {}
        self.official = official
        self.special_ids = special_ids
        self.official.requires_grad_(False)
        self.official.deberta.requires_grad_(True)
        self.projection = nn.Sequential(
            nn.Linear(official.config.hidden_size, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.evidence_head = nn.Linear(width, 3)
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.sentence_transformer = nn.TransformerEncoder(
            layer, 2, enable_nested_tensor=False
        )
        for layer in self.sentence_transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        if not self.use_sentence_interaction:
            del self.sentence_transformer

    def forward(self, blocks, n_sentences, labels=None):
        device = next(self.parameters()).device
        counts = torch.zeros(n_sentences, device=device)
        hidden_sums = None
        for block in blocks:
            ids = torch.tensor([block.input_ids], device=device)
            owners = torch.tensor(block.sentence_indices, device=device)
            valid = owners >= 0
            for special in self.special_ids:
                valid &= ids[0] != special
            hidden = self.official.deberta(
                input_ids=ids, attention_mask=torch.ones_like(ids)
            )[0][0]
            counts = counts + pool_sum(
                torch.ones(len(owners), device=device), owners, valid, n_sentences
            )
            values = pool_sum(hidden.float(), owners, valid, n_sentences)
            hidden_sums = values if hidden_sums is None else hidden_sums + values
        if not torch.all(counts > 0):
            raise ValueError("Missing sentence body tokens")
        features = self.projection(hidden_sums / counts[:, None])
        if self.use_sentence_interaction:
            features = self.sentence_transformer(features[None])[0]
        sentence_logits = self.evidence_head(features)
        self.last_probabilities = sentence_logits.float().softmax(-1).detach()
        scores = (
            sentence_logits.float().softmax(-1)
            * torch.tensor([0.0, 1.0 / 3.0, 1.0], device=device)
        ).sum(-1)
        if not labels:
            return (scores, None, counts)
        positives = [i for (i, label) in labels.items() if label == 2]
        negatives = [i for (i, label) in labels.items() if label == 0]
        if not positives or not negatives:
            raise ValueError("Query requires Direct and sampled qrel0 negatives")
        losses = {}
        supervised = {i: y for (i, y) in labels.items() if y != 1}
        losses["ternary"] = balanced_sentence_ce(
            sentence_logits, list(supervised), list(supervised.values())
        )
        loss = losses["ternary"]
        self.last_losses = {k: v.detach() for (k, v) in losses.items()}
        return (scores, loss, counts)


def optimizer_groups(model, base_lr=3e-06):
    """Published models use the same learning rates; each trainable parameter appears once."""
    (encoder, heads) = ([], [])
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("official.deberta.") else heads).append(
                parameter
            )
    return [
        {"params": encoder, "lr": base_lr, "name": "encoder"},
        {"params": heads, "lr": base_lr * 10, "name": "new_modules"},
    ]
