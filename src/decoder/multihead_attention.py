import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        mask: [seq_len, seq_len]
        """

        # Step 1: Linear projections
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        batch_size = query.shape[0]

        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]

        # Step 2: Split into heads
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim)

        # Step 3: Transpose for attention
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len_q, head_dim]
        K = K.transpose(1, 2)  # [batch, heads, seq_len_k, head_dim]
        V = V.transpose(1, 2)  # [batch, heads, seq_len_v, head_dim]

        # Step 4: Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 5: Apply mask
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        # Step 6: Weighted sum
        out = torch.matmul(attention, V)

        # Step 7: Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len_q, self.embed_dim)

        # Step 8: Final linear layer
        out = self.fc_out(out)

        return out