import os
import torch
import pandas as pd
from PIL import Image
from collections import defaultdict

from inference import load_models, generate_caption
from evaluation.bleu import compute_bleu
from src.data.transforms import image_transform

# =========================
# CONFIG
# =========================
DEBUG = True          # Print one sample
MAX_SAMPLES = None    # Set to 500 for testing, None for full run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

models = load_models()
models["encoder"].eval()
models["decoder"].eval()

print("Model loaded successfully!")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

caption_path = os.path.join(BASE_DIR, "data", "flickr8k", "captions.txt")
image_dir = os.path.join(BASE_DIR, "data", "flickr8k", "images")

# =========================
# LOAD DATA
# =========================
print("\nLoading dataset...")

df = pd.read_csv(caption_path)

df["clean_caption"] = df["caption"].str.lower()
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

print("Total caption rows:", len(df))

# =========================
# GROUP CAPTIONS BY IMAGE
# =========================
print("\nGrouping captions per image...")

image_caption_map = defaultdict(list)

for _, row in df.iterrows():
    image_caption_map[row["image"]].append(row["clean_caption"])

image_list = list(image_caption_map.keys())

print("Total unique images:", len(image_list))

# =========================
# CLEAN FUNCTION
# =========================
def clean_caption(caption):
    tokens = caption.split()
    tokens = [t for t in tokens if t not in ["<start>", "<end>", "<pad>"]]
    return " ".join(tokens)

# =========================
# EVALUATION
# =========================
print("\nStarting evaluation...\n")

references = []
predictions = []

count = 0

with torch.no_grad():
    for img_name in image_list:

        image_path = os.path.join(image_dir, img_name)

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

        image = image_transform(image).unsqueeze(0).to(device)

        # Generate caption
        pred_caption = generate_caption(image, models)
        pred_clean = clean_caption(pred_caption)
        pred_tokens = pred_clean.split()

        # Get all 5 captions
        captions_list = image_caption_map[img_name]

        ref_tokens = []
        for cap in captions_list:
            cap_clean = clean_caption(cap)
            ref_tokens.append(cap_clean.split())

        # DEBUG PRINT (only once)
        if DEBUG:
            print("------ SAMPLE DEBUG ------")
            print("Image:", img_name)
            print("Raw Prediction:", pred_caption)
            print("Clean Prediction:", pred_clean)
            print("Prediction Tokens:", pred_tokens)

            print("\nReferences:")
            for ref in ref_tokens:
                print(ref)

            print("\nImage tensor shape:", image.shape)
            print("--------------------------\n")

            DEBUG = False

        predictions.append(pred_tokens)
        references.append(ref_tokens)

        count += 1

        # Progress print
        if count % 50 == 0:
            print(f"Processed {count} images")

        # Optional limit for testing
        if MAX_SAMPLES and count >= MAX_SAMPLES:
            break

# =========================
# FINAL CHECK
# =========================
print("\n--- FINAL CHECK ---")
print("Total evaluated samples:", len(predictions))
print("Sample prediction:", predictions[0])
print("Sample references:", references[0])

# =========================
# COMPUTE BLEU
# =========================
print("\nComputing BLEU scores...\n")

scores = compute_bleu(references, predictions)

print("===== BLEU SCORES =====")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

# =========================
# SAVE RESULTS
# =========================
import json

with open("bleu_score.json", "w") as f:
    json.dump(scores, f)

print("\nBLEU scores saved to bleu_score.json")