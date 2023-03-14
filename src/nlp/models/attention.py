import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d, dropout=0.1):
        super(Attention, self).__init__()
        self.d = d
        self.to_qkv = nn.Linear(d, d * 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x has shape (B, T, d)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (B, T, d), (B, T, d), (B, T, d)
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # (B, T, T)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))  # (B, T, T)

        attn = attn.softmax(dim=-1)  # (B, T, T)
        return self.dropout(attn @ v)  # (B, T, T) @ (B, T, d) = (B, T, d)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, D, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert D % n_heads == 0, f"Cannot split dimension {D} into {n_heads} heads"

        self.D = D
        self.d = D // n_heads
        self.n_heads = n_heads

        self.heads = nn.ModuleList([Attention(self.d, dropout) for _ in range(n_heads)])

    def forward(self, x, mask=None):
        # x has shape (B, T, D)
        heads_out = [head(x[:, :, i * self.d: (i + 1) * self.d], mask) for i, head in enumerate(self.heads)]
        return torch.cat(heads_out, dim=-1)  # (B, T, D)
