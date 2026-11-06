from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .sentence_interaction import GatedResidualSentenceInteraction


@dataclass
class SetScores:
    sentence_logits: list[torch.Tensor]
    sentence_scores: list[torch.Tensor]
    document_logits: torch.Tensor
    document_scores: torch.Tensor


def _balanced_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    terms = []
    for label in torch.unique(targets):
        chosen = targets == label
        terms.append(F.cross_entropy(logits[chosen].float(), targets[chosen]))
    return torch.stack(terms).mean()


class HierarchicalProvence(nn.Module):
    """No-PE query-conditioned sentence/document set scorer.

    All four development variants construct the same modules in the same order.
    Flags only control which residual interactions are active, so differences
    cannot be attributed to different shared-head initialization.
    """

    def __init__(
        self,
        official: nn.Module,
        special_ids: list[int],
        *,
        use_sentence_interaction: bool,
        use_document_interaction: bool,
        width: int = 256,
    ):
        super().__init__()
        self.official = official
        self.special_ids = list(special_ids)
        self.width = width
        self.use_sentence_interaction = use_sentence_interaction
        self.use_document_interaction = use_document_interaction
        self.use_sentence_position_encoding = False

        self.official.requires_grad_(False)
        self.official.deberta.requires_grad_(True)
        self.projection = nn.Sequential(
            nn.Linear(official.config.hidden_size, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.evidence_head = nn.Linear(width, 3)

        sentence_layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        sentence_transformer = nn.TransformerEncoder(
            sentence_layer, 2, enable_nested_tensor=False
        )
        for layer in sentence_transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.sentence_interaction = GatedResidualSentenceInteraction(
            sentence_transformer, width, gate_init=0.0
        )

        self.document_pool = nn.Linear(width, 1)
        document_layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        document_transformer = nn.TransformerEncoder(
            document_layer, 1, enable_nested_tensor=False
        )
        for layer in document_transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.document_interaction_norm = nn.LayerNorm(width)
        self.document_interaction = document_transformer
        self.document_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.document_to_sentence = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.feedback_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.document_head = nn.Linear(width, 1)

    def _encode_prepared(
        self, prepared, batch_size: int, pad_id: int
    ) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        hidden_width = self.official.config.hidden_size
        sums = [
            torch.zeros((len(ordered), hidden_width), device=device)
            for ordered, _ in prepared
        ]
        counts = [torch.zeros(len(ordered), device=device) for ordered, _ in prepared]
        jobs = [
            (doc_index, block.input_ids, block.sentence_indices)
            for doc_index, (_, blocks) in enumerate(prepared)
            for block in blocks
        ]
        for begin in range(0, len(jobs), batch_size):
            batch = jobs[begin : begin + batch_size]
            token_width = max(len(ids) for _, ids, _ in batch)
            input_ids = torch.full(
                (len(batch), token_width), pad_id, dtype=torch.long, device=device
            )
            owners = torch.full_like(input_ids, -1)
            mask = torch.zeros_like(input_ids)
            for row, (_, ids, sentence_indices) in enumerate(batch):
                length = len(ids)
                input_ids[row, :length] = torch.tensor(ids, device=device)
                owners[row, :length] = torch.tensor(sentence_indices, device=device)
                mask[row, :length] = 1
            hidden = self.official.deberta(input_ids=input_ids, attention_mask=mask)[
                0
            ].float()
            for row, (doc_index, _, _) in enumerate(batch):
                valid = (owners[row] >= 0) & mask[row].bool()
                for special in self.special_ids:
                    valid &= input_ids[row] != special
                selected = owners[row, valid]
                sums[doc_index].index_add_(0, selected, hidden[row, valid])
                counts[doc_index].index_add_(
                    0, selected, torch.ones(len(selected), device=device)
                )
        if any(not torch.all(count > 0) for count in counts):
            raise ValueError("Missing sentence body tokens")
        return [
            self.projection(total / count[:, None])
            for total, count in zip(sums, counts, strict=True)
        ]

    def _score_features(self, features: list[torch.Tensor]) -> SetScores:
        if self.use_sentence_interaction:
            features = [self.sentence_interaction(value[None])[0] for value in features]

        document_vectors = []
        for value in features:
            attention = F.softmax(self.document_pool(value).squeeze(-1), dim=0)
            document_vectors.append((attention[:, None] * value).sum(0))
        documents = torch.stack(document_vectors)

        if self.use_document_interaction:
            delta = self.document_interaction(
                self.document_interaction_norm(documents)[None]
            )[0]
            documents = documents + torch.tanh(self.document_gate_logit) * delta
            conditioned = []
            feedback_gate = torch.tanh(self.feedback_gate_logit)
            for value, document in zip(features, documents, strict=True):
                context = document.expand(len(value), -1)
                feedback = self.document_to_sentence(
                    torch.cat([value, context], dim=-1)
                )
                conditioned.append(value + feedback_gate * feedback)
            features = conditioned

        logits = [self.evidence_head(value) for value in features]
        weights = torch.tensor([0.0, 1.0 / 3.0, 1.0], device=documents.device)
        sentence_scores = [
            (F.softmax(value.float(), dim=-1) * weights).sum(-1) for value in logits
        ]
        document_logits = self.document_head(documents).squeeze(-1)
        return SetScores(
            sentence_logits=logits,
            sentence_scores=sentence_scores,
            document_logits=document_logits,
            document_scores=torch.sigmoid(document_logits.float()),
        )

    def forward(self, prepared, batch_size: int, pad_id: int) -> SetScores:
        return self._score_features(self._encode_prepared(prepared, batch_size, pad_id))

    def loss(
        self,
        scores: SetScores,
        sentence_labels: list[list[int]],
        document_labels: list[int],
        document_loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sentence_logits, sentence_targets = [], []
        for logits, labels in zip(scores.sentence_logits, sentence_labels, strict=True):
            for index, label in enumerate(labels):
                if label != 1:  # Partial remains excluded, never converted to negative.
                    sentence_logits.append(logits[index])
                    sentence_targets.append(label)
        target = torch.tensor(
            sentence_targets, dtype=torch.long, device=scores.document_logits.device
        )
        logits = torch.stack(sentence_logits)
        sentence_loss = _balanced_ce(logits, target)

        doc_target = torch.tensor(
            document_labels, dtype=torch.float32, device=scores.document_logits.device
        )
        pos = doc_target == 1
        neg = doc_target == 0
        if not pos.any() or not neg.any():
            raise ValueError(
                "Document-set loss requires positive and explicit negative documents"
            )
        document_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                scores.document_logits[pos].float(), doc_target[pos]
            )
            + F.binary_cross_entropy_with_logits(
                scores.document_logits[neg].float(), doc_target[neg]
            )
        )
        total = sentence_loss + document_loss_weight * document_loss
        return total, {
            "sentence_ce": float(sentence_loss.detach()),
            "document_bce": float(document_loss.detach()),
            "sentence_gate": float(self.sentence_interaction.gate_logit.detach()),
            "document_gate": float(self.document_gate_logit.detach()),
            "feedback_gate": float(self.feedback_gate_logit.detach()),
        }
