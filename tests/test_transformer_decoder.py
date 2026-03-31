import torch
from src.decoder.transformer_decoder import TransformerDecoder

batch_size = 16
seq_len = 41
embed_dim = 256
vocab_size = 8000

x = torch.randn(batch_size, seq_len, embed_dim)
encoder_output = torch.randn(batch_size, 1, embed_dim)

decoder = TransformerDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=8,
    num_layers=2,
    max_len=100
)

out = decoder(x, encoder_output)

print("Input:", x.shape)
print("Output:", out.shape)