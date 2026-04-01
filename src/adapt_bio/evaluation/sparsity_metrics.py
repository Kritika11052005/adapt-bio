"""sparsity_metrics.py — measures SOMA mask sparsity stats."""
import torch
from typing import Dict

def compute_sparsity_stats(mask: torch.Tensor) -> Dict[str, float]:
    total = mask.numel()
    active = mask.sum().item()
    return {
        "overall_sparsity": 1.0 - active / total,
        "effective_k_mean": mask.float().sum(dim=-1).mean().item(),
    }
