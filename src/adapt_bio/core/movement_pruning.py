import torch
import torch.nn as nn
from .starling_topology import StarlingTopologyConstraint

class MovementPruningMask(nn.Module):
    def __init__(self, num_heads, seq_len, anticipation_steps):
        super().__init__()
        self.anticipation_steps = anticipation_steps
        self.step = 0
        self.register_buffer("movement_accum", torch.zeros(num_heads, seq_len, seq_len))

    def update(self, attn_weights):
        if self.step == 0:
            self.register_buffer("w0", attn_weights.detach().clone())
        if self.step < self.anticipation_steps:
            self.movement_accum += (attn_weights.detach() - self.w0).abs()
        self.step += 1

    def emit_early_mask(self, k=7):
        return StarlingTopologyConstraint(k=k).apply(self.movement_accum)
