"""trainer.py — ADAPT-BIO training loop."""
class ADAPTBIOTrainer:
    def __init__(self, model, optimizer, scheduler, config: dict, wandb_run=None):
        self.model = model
        self.config = config
        self.step = 0
    def train_step(self, batch) -> dict:
        raise NotImplementedError
    def run(self, dataloader, max_steps: int):
        raise NotImplementedError
