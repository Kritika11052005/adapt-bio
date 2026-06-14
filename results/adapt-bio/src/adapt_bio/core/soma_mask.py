import torch
import torch.nn as nn
from .movement_pruning import MovementPruningMask
from .rna_mask_refinement import RNAMaskRefinement
from .starling_topology import StarlingTopologyConstraint

class SOMAMask(nn.Module):
    def __init__(self, num_heads, seq_len, k=7, anticipation_steps=100, rna_update_interval=500):
        super().__init__()
        self.movement = MovementPruningMask(num_heads, seq_len, anticipation_steps)
        self.rna = RNAMaskRefinement(update_interval=rna_update_interval)
        self.starling = StarlingTopologyConstraint(k=k)
        self.register_buffer("current_mask", torch.ones(num_heads, seq_len, seq_len, dtype=torch.bool))

    def forward(self, attn_weights, step):
        self.movement.update(attn_weights)
        self.current_mask = self.rna.refine(
            self.current_mask, self.movement.movement_accum, step=step, k=self.starling.k
        )
        return self.current_mask
