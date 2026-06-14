import torch
import torch.nn as nn

class MHCResidual(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x, sublayer_output):
        normed = self.norm(x + sublayer_output)
        return normed * torch.sigmoid(self.scale)
