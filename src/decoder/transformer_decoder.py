import torch
import torch.nn as nn
from src.decoder.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_output, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        encoder_output: [batch, 1, embed_dim]
        """

        for layer in self.layers:
            x = layer(x, encoder_output, mask)

        out = self.fc_out(x)  # [batch, seq_len, vocab_size]

        return out