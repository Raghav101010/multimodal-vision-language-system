from evaluation.bleu import compute_bleu
from inference import load_models, generate_caption
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
from src.data.dataset import CaptionDataset
from src.data.tokenizer import InferenceTokenizer, Tokenizer
from src.data.transforms import image_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

tokenizer = InferenceTokenizer(word2idx, idx2word)

def decode_tokens(token_ids, idx2word):
    words = []
    for idx in token_ids:
        word = idx2word.get(int(idx), "")
        if word not in ["<start>", "<end>", "<pad>"]:
            words.append(word)
    return " ".join(words)

config = {
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "max_len": 100,
    "vocab_size": len(tokenizer.word2idx),
}

models = load_models()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

caption_path = os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt")
image_dir = os.path.join(BASE_DIR, "data", "flickr8k", "images")

df = pd.read_csv(caption_path)

df["clean_caption"] = df["caption"].str.lower()
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

token = Tokenizer(df["clean_caption"])
dataset = CaptionDataset(df, token, image_dir, image_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

references = []
predictions = []

models["encoder"].eval()
models["decoder"].eval()

def clean_caption(caption):
    return caption.replace("<start>", "").replace("<end>", "").strip()

print("Dataset size:", len(dataset))
print("Sample image dir:", image_dir)
print("Number of images found:", len(os.listdir(image_dir)))
count = 0
with torch.no_grad():
    debug = True
    for images, captions in dataloader:
        for i in range(min(len(images), 5)):
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} samples")
            image = images[i].unsqueeze(0).to(device)

            pred_caption = generate_caption(image, models)

            pred_tokens = clean_caption(pred_caption).split()
            #ref_tokens = [clean_caption(captions[i]).split()]  # assuming 1 caption per image
            ref_caption = decode_tokens(captions[i], idx2word)
            ref_tokens = [ref_caption.split()]

            # DEBUG PRINTS
            if debug:
                print("\n--- SAMPLE DEBUG ---", i)
                print("Raw prediction:", pred_caption)
                #print("Clean prediction:", pred_clean)
                print("Prediction tokens:", pred_tokens)

                print("Decoded reference:", ref_caption)
                print("Reference tokens:", ref_tokens)

                print("Image tensor shape:", image.shape)

                debug = False  # print only once

            predictions.append(pred_tokens)
            references.append(ref_tokens)

print("\n--- FINAL CHECK ---")
print("Total samples:", len(predictions))
print("Sample prediction:", predictions[0])
print("Sample reference:", references[0])

scores = compute_bleu(references, predictions)

print(scores)