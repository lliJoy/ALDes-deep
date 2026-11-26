import torch
from torch import nn


class FeatureWiseModulation(nn.Module):
    """Apply bounded FiLM conditioning to every sequence position."""

    def __init__(self, context_dim, d_model, gamma_scale=0.1):
        super().__init__()
        if context_dim <= 0 or d_model <= 0:
            raise ValueError("FiLM dimensions must be positive.")
        if gamma_scale <= 0:
            raise ValueError("FiLM gamma_scale must be positive.")
        self.gamma_scale = float(gamma_scale)
        self.modulation = nn.Linear(context_dim, 2 * d_model)
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, sequence, context):
        if context is None or context.ndim != 2:
            raise ValueError("FiLM context must have shape (batch, context_dim).")
        if context.shape[0] == 1 and sequence.shape[0] > 1:
            context = context.expand(sequence.shape[0], -1)
        if context.shape[0] != sequence.shape[0]:
            raise ValueError("FiLM context batch must match the sequence batch.")
        gamma, beta = self.modulation(context).chunk(2, dim=-1)
        gamma = self.gamma_scale * torch.tanh(gamma)
        return sequence * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
