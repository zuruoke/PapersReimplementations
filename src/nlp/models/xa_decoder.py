import torch.nn as nn
from src.nlp.models.attention import MultiHeadAttention


class CrossAttentionDecoder(nn.Module):
    def __init__(self, n_heads, dim, dropout=0.1, mlp_scale=4, activation=nn.GELU):
        super(CrossAttentionDecoder, self).__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.mlp_scale = mlp_scale
        self.activation = activation

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mha = MultiHeadAttention(n_heads, dim, dropout)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.cross_mha = MultiHeadAttention(n_heads, dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_scale),
            activation(),
            nn.Linear(dim * mlp_scale, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z, mask=None):
        # Masked multi-head self attention
        q = x + self.mha(self.ln1(x), mask)
        k, v = self.to_kv(z).chunk(2, dim=-1)

        # Cross attention
        att = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        att = att.softmax(dim=-1)
        out = att @ v

        # Feed forward
        out = out + self.mlp(self.ln2(out))
        return out
