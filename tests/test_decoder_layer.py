import torch
from src.decoder.decoder_layer import DecoderLayer
from src.decoder.utils import generate_causal_mask

x = torch.randn(16, 41, 256)
seq_len = x.shape[1]
mask = generate_causal_mask(seq_len)
encoder_output = torch.randn(16, 1, 256)

layer = DecoderLayer(embed_dim=256, num_heads=8)

out = layer(x, encoder_output, mask=mask)

print("Input:", x.shape)
print("Encoder Output:", encoder_output.shape)
print("Output:", out.shape)
