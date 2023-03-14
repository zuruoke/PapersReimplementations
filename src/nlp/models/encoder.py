import torch.nn as nn
from src.nlp.models.attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1, mlp_scale=4, activation=nn.GELU):
        super(Encoder, self).__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mha = MultiHeadAttention(n_heads, dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_scale),
            activation(),
            nn.Linear(dim * mlp_scale, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x has shape (B, T, D)
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderBlocks(nn.Module):
    def __init__(self, n_layers, dim, n_heads, dropout=0.1, mlp_scale=4, activation=nn.GELU):
        super(EncoderBlocks, self).__init__()

        self.layers = nn.ModuleList([
            Encoder(dim, n_heads, dropout, mlp_scale, activation)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x has shape (B, T, D)
        for layer in self.layers:
            x = layer(x)
        return x
