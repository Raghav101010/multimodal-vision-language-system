import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.data.dataset import CaptionDataset
from src.data.tokenizer import Tokenizer
from src.data.transforms import image_transform


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

caption_path = os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt")
image_dir = os.path.join(BASE_DIR, "data", "flickr8k", "images")

df = pd.read_csv(caption_path)

df["clean_caption"] = df["caption"].str.lower()
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

tokenizer = Tokenizer(df["clean_caption"])

dataset = CaptionDataset(df, tokenizer, image_dir, image_transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Test one batch
images, captions = next(iter(dataloader))

print("Images:", images.shape)
print("Captions:", captions.shape)
print(len(dataloader))

# Test encoder with real data
from src.encoder.encoder import EncoderCNN

encoder = EncoderCNN(embed_size=256)
features = encoder(images)
print("Features:", features.shape)