import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]

        # Register as buffer (NOT a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)

        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]

        return x
    
