import torch
import torch.nn as nn
from src.decoder.multihead_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=1024, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        encoder_output: [batch, 1, embed_dim]
        """

        # 1 Masked Self Attention
        _x = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(_x))

        # 2 Cross Attention (Image)
        _x = self.cross_attn(x, encoder_output, encoder_output)  
        x = self.norm2(x + self.dropout(_x))

        # 3 Feed Forward
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout(_x))

        return x