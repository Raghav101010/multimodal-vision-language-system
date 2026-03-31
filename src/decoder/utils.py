import torch

def generate_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # [seq_len, seq_len]