"""
sparse_attention.py
-------------------
Standard scaled dot-product attention, modified to accept a SOMA sparse mask.
The mask is computed BEFORE attention (at compile time for hardware accelerators).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..core.soma_mask import SOMAMask


class SOMAAttention(nn.Module):
    """Multi-head attention with SOMA sparse mask."""

    def __init__(self, d_model: int, num_heads: int, seq_len: int, soma_cfg: dict):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.soma = SOMAMask(**soma_cfg)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        raise NotImplementedError
