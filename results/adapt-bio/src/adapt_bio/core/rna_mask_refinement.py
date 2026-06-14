import torch
import torch.nn as nn
from .starling_topology import StarlingTopologyConstraint

class RNAMaskRefinement(nn.Module):
    def __init__(self, update_interval=500):
        super().__init__()
        self.update_interval = update_interval
        self.last_update_step = 0

    def should_update(self, step):
        return (step - self.last_update_step) >= self.update_interval

    def refine(self, current_mask, movement_signal, step, k=7):
        if not self.should_update(step):
            return current_mask
        new_mask = StarlingTopologyConstraint(k=k).apply(movement_signal)
        self.last_update_step = step
        return new_mask
