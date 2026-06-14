import torch
import torch.nn as nn
from ..attention.sparse_attention import SOMAAttention
from ..core.mhc_residual import MHCResidual

class ADAPTBIOBlock(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, ffn_mult=4, k=7,
                 anticipation_steps=10, rna_update_interval=5):
        super().__init__()
        self.attn = SOMAAttention(d_model, num_heads, seq_len, k=k,
                                  anticipation_steps=anticipation_steps,
                                  rna_update_interval=rna_update_interval)
        self.mhc1 = MHCResidual(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model)
        )
        self.mhc2 = MHCResidual(d_model)

    def forward(self, x, step):
        x = self.mhc1(x, self.attn(x, step))
        x = self.mhc2(x, self.ffn(x))
        return x

class ADAPTBIOTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 seq_len, k=7, anticipation_steps=10, rna_update_interval=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            ADAPTBIOBlock(d_model, num_heads, seq_len, k=k,
                          anticipation_steps=anticipation_steps,
                          rna_update_interval=rna_update_interval)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, step):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.embed(token_ids) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x, step)
        return self.lm_head(self.norm(x))
