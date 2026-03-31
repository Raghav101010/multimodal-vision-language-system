import torch
from src.decoder.multihead_attention import MultiHeadAttention

x = torch.randn(16, 41, 256)

mha = MultiHeadAttention(embed_dim=256, num_heads=8)

out = mha(x)

print("Input:", x.shape)
print("Output:", out.shape)