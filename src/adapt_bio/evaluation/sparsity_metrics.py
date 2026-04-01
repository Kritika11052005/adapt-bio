
import torch
import numpy as np
from typing import Dict, List

def compute_sparsity(mask: torch.Tensor) -> float:
    """Fraction of zero (inactive) attention edges."""
    return 1.0 - mask.float().mean().item()

def compute_edges_per_token(mask: torch.Tensor) -> float:
    """Average number of active attention edges per token."""
    # mask shape: (B, H, T, T) or (B, T, T)
    if mask.dim() == 4:
        # sum over last dim = active neighbors per token
        return mask.float().sum(dim=-1).mean().item()
    return mask.float().sum(dim=-1).mean().item()

def compute_sparsity_over_time(sparsity_log: List[float]) -> Dict:
    """Summary stats from a list of per-step sparsity values."""
    arr = np.array(sparsity_log)
    return {
        "mean":    float(arr.mean()),
        "std":     float(arr.std()),
        "min":     float(arr.min()),
        "max":     float(arr.max()),
        "final":   float(arr[-1]),
        "steps":   len(arr),
    }

def compute_theoretical_flop_reduction(k: int, seq_len: int) -> Dict:
    """
    Theoretical FLOPs saved by ADAPT-BIO sparse attention vs dense.
    Dense attention: O(T^2), Sparse: O(T*k)
    """
    dense_flops  = seq_len * seq_len
    sparse_flops = seq_len * k
    reduction    = 1.0 - sparse_flops / dense_flops
    return {
        "dense_attn_flops":  dense_flops,
        "sparse_attn_flops": sparse_flops,
        "flop_reduction":    reduction,
        "edges_active_pct":  (1 - reduction) * 100,
    }

def sparsity_report(mask: torch.Tensor, k: int, seq_len: int, step: int = 0) -> Dict:
    """Full sparsity report for a single mask tensor."""
    sparsity   = compute_sparsity(mask)
    edges      = compute_edges_per_token(mask)
    flop_stats = compute_theoretical_flop_reduction(k, seq_len)
    return {
        "step":              step,
        "sparsity":          sparsity,
        "edges_per_token":   edges,
        "flop_reduction":    flop_stats["flop_reduction"],
        "edges_active_pct":  flop_stats["edges_active_pct"],
        "dense_flops":       flop_stats["dense_attn_flops"],
        "sparse_flops":      flop_stats["sparse_attn_flops"],
    }
