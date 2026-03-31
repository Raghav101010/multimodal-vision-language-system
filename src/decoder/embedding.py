import torch
import torch.nn as nn


class DecoderInput(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, image_features, captions):
        """
        image_features: [batch, embed_dim]
        captions: [batch, seq_len]
        """

        # Convert captions to embeddings
        caption_embeddings = self.embedding(captions)
        # [batch, seq_len, embed_dim]

        # Convert image to token
        image_features = image_features.unsqueeze(1)
        # [batch, 1, embed_dim]

        # Concatenate
        x = torch.cat((image_features, caption_embeddings), dim=1)
        # [batch, seq_len+1, embed_dim]

        return x