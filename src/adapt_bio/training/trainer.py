
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import os

class WikiText2Dataset(torch.utils.data.Dataset):
    def __init__(self, split="train", seq_len=128, tokenizer_name="gpt2"):
        self.seq_len = seq_len
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join([x for x in dataset["text"] if x.strip()])
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze()
        n = (len(tokens) - 1) // seq_len
        self.inputs = tokens[:n * seq_len].view(n, seq_len)
        self.targets = tokens[1:n * seq_len + 1].view(n, seq_len)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ADAPTBIOTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get("max_steps", 5000)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_perplexities = []
        self.sparsity_log = []
        self.global_step = 0

    def compute_perplexity(self, loss):
        return torch.exp(torch.tensor(loss)).item()

    def get_sparsity(self):
        total, sparse = 0, 0
        for module in self.model.modules():
            if hasattr(module, "soma") and hasattr(module.soma, "current_mask"):
                mask = module.soma.current_mask
                if mask is not None:
                    total += mask.numel()
                    sparse += (mask == 0).sum().item()
        return sparse / total if total > 0 else 0.0

    def train_step(self, inputs, targets):
        self.model.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(inputs, step=self.global_step)   # ← pass step
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return loss.item()

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches=50):
        self.model.eval()
        total_loss, count = 0.0, 0
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= max_batches:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs, step=999999)   # ← large step = fully sparse
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / count
        return avg_loss, self.compute_perplexity(avg_loss)

    def train(self, train_loader, val_loader, max_steps=5000, eval_every=250, log_every=50):
        print(f"Training on: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Max steps: {max_steps} | Eval every: {eval_every}\n")
        start = time.time()
        train_iter = iter(train_loader)
        for step in range(max_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
            loss = self.train_step(inputs, targets)
            self.train_losses.append(loss)
            if step % log_every == 0:
                sparsity = self.get_sparsity()
                self.sparsity_log.append((step, sparsity))
                elapsed = time.time() - start
                print(f"Step {step:5d} | Loss: {loss:.4f} | PPL: {self.compute_perplexity(loss):.2f} | Sparsity: {sparsity:.1%} | {elapsed:.0f}s")
            if step % eval_every == 0 and step > 0:
                val_loss, val_ppl = self.evaluate(val_loader)
                self.val_perplexities.append((step, val_ppl))
                print(f"\n{'='*55}")
                print(f"  EVAL @ step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                print(f"{'='*55}\n")
        print("\nTraining complete.")
        return self.train_losses, self.val_perplexities, self.sparsity_log


def run_experiment(model, config):
    print("Loading WikiText-2...")
    train_ds = WikiText2Dataset(split="train",      seq_len=config["seq_len"])
    val_ds   = WikiText2Dataset(split="validation", seq_len=config["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=2)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    trainer = ADAPTBIOTrainer(model, config)
    return trainer.train(
        train_loader, val_loader,
        max_steps=config.get("max_steps", 5000),
        eval_every=config.get("eval_every", 250),
        log_every=config.get("log_every", 50),
    )
