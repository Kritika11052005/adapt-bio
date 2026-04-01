"""
rna_mask_refinement.py
----------------------
Octopus RNA-editing insight: the mask is NOT frozen after initial discovery.
Every `update_interval` steps, movement signals re-evaluate the mask.
This enables mid-training topology correction — novel vs all prior pruning work.
"""
import torch
import torch.nn as nn


class RNAMaskRefinement(nn.Module):
    """
    Periodically re-edits the sparse mask based on fresh movement signals.
    Analogous to octopus mRNA editing: rewrites active code while running.
    """

    def __init__(self, update_interval: int = 500):
        super().__init__()
        self.update_interval = update_interval
        self.last_update_step = 0

    def should_update(self, step: int) -> bool:
        return (step - self.last_update_step) >= self.update_interval

    def refine(self, current_mask: torch.Tensor, movement_signal: torch.Tensor) -> torch.Tensor:
        """Returns updated mask. Override with task-specific logic."""
        raise NotImplementedError
