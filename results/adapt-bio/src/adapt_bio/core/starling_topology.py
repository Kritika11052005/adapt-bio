import torch
import torch.nn as nn

class StarlingTopologyConstraint(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.k = k

    def apply(self, movement_scores: torch.Tensor) -> torch.Tensor:
        k = min(self.k, movement_scores.shape[-1])
        topk_vals, _ = movement_scores.topk(k, dim=-1)
        threshold = topk_vals[..., -1].unsqueeze(-1)
        return movement_scores >= threshold
