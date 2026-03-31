import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.data.dataset import CaptionDataset
from src.data.tokenizer import Tokenizer
from src.data.transforms import image_transform
from src.encoder.encoder import EncoderCNN
from src.decoder.embedding import DecoderInput
from src.decoder.positional_encoding import PositionalEncoding


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

caption_path = os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt")
image_dir = os.path.join(BASE_DIR, "data", "flickr8k", "images")

df = pd.read_csv(caption_path)

df["clean_caption"] = df["caption"].str.lower()
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

tokenizer = Tokenizer(df["clean_caption"])

dataset = CaptionDataset(df, tokenizer, image_dir, image_transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

encoder = EncoderCNN(embed_size=256)
encoder.eval()
decoder = DecoderInput(vocab_size = len(tokenizer.word2idx), embed_dim = 256)
pe = PositionalEncoding(embed_dim=256, max_len=100)

#images, captions = next(iter(dataloader))
images, captions = next(iter(dataloader))
# Encode the images into 256 dimensions vector
features = encoder(images)
decoder_input = decoder(features, captions)
out = pe(decoder_input)

print("Images:", images.shape)
print("Captions:", captions.shape)

print("Encoded features:", features.shape)
print("Decoder input:", decoder_input.shape)
print("After Positional Encoding:", out.shape)