import torch
from src.encoder.encoder import EncoderCNN

# Create dummy image batch
images = torch.randn(4, 3, 224, 224)

encoder = EncoderCNN(embed_size=256)

features = encoder(images)

print("Input shape:", images.shape)
print("Feature shape:", features.shape)