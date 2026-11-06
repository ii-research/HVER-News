from __future__ import annotations

import torch
from torch import nn


class GatedResidualSentenceInteraction(nn.Module):
    """Turn the existing sentence transformer into a safe residual update.

    A zero-initialized gate makes the module an exact identity at step zero.
    Consequently, HVER-News starts from the same sentence features as
    the no-SI/no-PE control and can only introduce interaction gradually.
    """

    def __init__(self, transformer: nn.Module, width: int, gate_init: float = 0.0):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.transformer = transformer
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta = self.transformer(self.input_norm(features))
        return features + torch.tanh(self.gate_logit) * delta
