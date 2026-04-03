import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.data.dataset import CaptionDataset
from src.data.tokenizer import Tokenizer
from src.data.transforms import image_transform

from src.encoder.encoder import EncoderCNN
from src.decoder.embedding import DecoderInput
from src.decoder.positional_encoding import PositionalEncoding
from src.decoder.transformer_decoder import TransformerDecoder
from src.decoder.utils import generate_causal_mask


# ===== Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/content"
caption_path = os.path.join(BASE_DIR, "flickr8k", "captions.txt")
image_dir = os.path.join(BASE_DIR, "flickr8k", "images")

SAVE_DIR = "/content/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===== Load Data =====
df = pd.read_csv(caption_path)

df["clean_caption"] = df["caption"].str.lower()
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

tokenizer = Tokenizer(df["clean_caption"], min_freq=2)

dataset = CaptionDataset(df, tokenizer, image_dir, image_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Dataset size:", len(dataset))
print("Dataloader batches:", len(dataloader))


# ===== Models =====
embed_dim = 256
vocab_size = len(tokenizer.word2idx)

encoder = EncoderCNN(embed_dim).to(device)
decoder_input = DecoderInput(vocab_size, embed_dim).to(device)
pos_enc = PositionalEncoding(embed_dim).to(device)

decoder = TransformerDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=8,
    num_layers=4,
    max_len=100
).to(device)


# ===== Loss & Optimizer =====
criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = torch.optim.Adam([
    {"params": decoder.parameters(), "lr": 1e-4},
    {"params": decoder_input.parameters(), "lr": 1e-4},
    {"params": encoder.resnet.layer4.parameters(), "lr": 1e-5}
])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)


# ===== Training Config =====
epochs = 20
best_loss = float("inf")

# Early stopping
patience = 3
no_improve_epochs = 0


# ===== Training Loop =====
for epoch in range(epochs):

    print(f"\n Starting Epoch {epoch+1}")
    epoch_loss = 0

    for i, (images, captions) in enumerate(dataloader):

        images = images.to(device)
        captions = captions.to(device)

        # ===== Forward =====
        features = encoder(images)

        input_seq = captions[:, :-1]
        target_seq = captions[:, 1:]

        x = decoder_input(features, input_seq)
        x = pos_enc(x)

        mask = generate_causal_mask(x.shape[1]).to(device)

        outputs = decoder(x, features.unsqueeze(1), mask)

        outputs = outputs[:, 1:, :]
        outputs = outputs.reshape(-1, vocab_size)
        target_seq = target_seq.reshape(-1)

        loss = criterion(outputs, target_seq)
        epoch_loss += loss.item()

        # ===== Backprop =====
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(decoder.parameters()) + list(decoder_input.parameters()),
            1.0
        )

        optimizer.step()

        if i % 200 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}", flush=True)

    # ===== Scheduler Step =====
    scheduler.step()

    epoch_avg_loss = epoch_loss / len(dataloader)
    print(f" Epoch {epoch+1}, Avg Loss: {epoch_avg_loss:.4f}")

    # ===== Save LAST model =====
    torch.save({
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'decoder_input': decoder_input.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': epoch_avg_loss,
        'word2idx': tokenizer.word2idx,
        'idx2word': tokenizer.idx2word
    }, os.path.join(SAVE_DIR, "last_checkpoint.pth"))

    # ===== Save BEST model =====
    if epoch_avg_loss < best_loss:
        best_loss = epoch_avg_loss
        no_improve_epochs = 0

        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'decoder_input': decoder_input.state_dict(),
            'word2idx': tokenizer.word2idx,
            'idx2word': tokenizer.idx2word
        }, os.path.join(SAVE_DIR, "best_model.pth"))

        print(f" Best model saved at epoch {epoch+1}")

    else:
        no_improve_epochs += 1
        print(f" No improvement for {no_improve_epochs} epoch(s)")

    # ===== Early Stopping =====
    if no_improve_epochs >= patience:
        print(" Early stopping triggered!")
        break


print("\n Training Complete!")