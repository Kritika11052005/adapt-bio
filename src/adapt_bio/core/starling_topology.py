"""
starling_topology.py
--------------------
Starling murmuration insight: each token attends to only its k most
movement-active neighbors. Global coherent representations emerge from
purely local sparse connections — and maps directly onto tensor core ops.
"""
import torch
import torch.nn as nn


class StarlingTopologyConstraint(nn.Module):
    """
    Enforces that each query token attends to at most k key tokens.
    k is the 'neighbor count' — default 7 mirrors the starling 7-neighbor rule.
    """

    def __init__(self, k: int = 7):
        super().__init__()
        self.k = k

    def apply(self, movement_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            movement_scores: (num_heads, seq_len, seq_len) movement magnitudes
        Returns:
            boolean mask where each row keeps only top-k entries
        """
        topk_vals, _ = movement_scores.topk(self.k, dim=-1)
        threshold = topk_vals[..., -1].unsqueeze(-1)
        return movement_scores >= threshold
