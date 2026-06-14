import torch
import torch.nn as nn
from ..core.soma_mask import SOMAMask

class SOMAAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, k=7,
                 anticipation_steps=100, rna_update_interval=50):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.soma = SOMAMask(num_heads=num_heads, seq_len=seq_len, k=k,
                             anticipation_steps=anticipation_steps,
                             rna_update_interval=rna_update_interval)

    def forward(self, x, step):
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        q, k_vec, v = [t.view(B, T, self.num_heads, self.d_k).transpose(1,2) for t in qkv]
        scores = (q @ k_vec.transpose(-2,-1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        soma_input = attn_weights.detach().mean(dim=0)
        mask = self.soma(soma_input, step=step)
        masked_attn = attn_weights * mask.unsqueeze(0).float()
        masked_attn = masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-9)
        out = (masked_attn @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)
