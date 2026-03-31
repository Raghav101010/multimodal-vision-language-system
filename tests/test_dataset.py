import os
import pandas as pd
from src.data.tokenizer import Tokenizer
from src.data.dataset import CaptionDataset
from src.data.transforms import image_transform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caption_path = os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt")
image_dir = os.path.join(BASE_DIR, "data", "flickr8k", "images")

df = pd.read_csv(caption_path)

df["clean_caption"] = df["caption"].str.lower()
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

tokenizer = Tokenizer(df["clean_caption"])

dataset = CaptionDataset(
    df,
    tokenizer,
    image_dir,
    image_transform
)

image, caption = dataset[0]

print("Image shape:", image.shape)
print("Caption shape:", caption.shape)
print("Caption:", caption)
print("Image", type(image))