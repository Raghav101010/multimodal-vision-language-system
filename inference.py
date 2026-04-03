import torch
from PIL import Image

from src.encoder.encoder import EncoderCNN
from src.decoder.embedding import DecoderInput
from src.decoder.positional_encoding import PositionalEncoding
from src.decoder.transformer_decoder import TransformerDecoder
from src.decoder.utils import generate_causal_mask
from src.data.transforms import image_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_pil(image):
    image = image.convert("RGB")
    image = image_transform(image)
    return image.unsqueeze(0).to(device)

def load_image_from_path(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess_pil(image)

# ----------------------------
# Load checkpoint + models
# ----------------------------
def load_models():
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)

    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]

    vocab_size = len(word2idx)
    embed_dim = 256

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

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    decoder_input.load_state_dict(checkpoint["decoder_input"])

    encoder.eval()
    decoder.eval()
    decoder_input.eval()

    return {
        "encoder": encoder,
        "decoder": decoder,
        "decoder_input": decoder_input,
        "pos_enc": pos_enc,
        "word2idx": word2idx,
        "idx2word": idx2word
    }

def generate_caption(image_input, models, beam=True):

    device = next(models["encoder"].parameters()).device

    if isinstance(image_input, torch.Tensor):
        # Already preprocessed (from DataLoader)
        image = image_input.to(device)

    elif isinstance(image_input, str):
        image = load_image_from_path(image_input).to(device)

    else:
        # PIL Image
        image = preprocess_pil(image_input).to(device)

    if beam:
        return beam_search(image, models)
    else:
        return greedy_decode(image, models)
# ----------------------------
# Greedy decoding (CORRECT)
# ----------------------------
def greedy_decode(image_tensor, models, max_len=30):
    encoder = models["encoder"]
    decoder = models["decoder"]
    decoder_input = models["decoder_input"]
    pos_enc = models["pos_enc"]
    word2idx = models["word2idx"]
    idx2word = models["idx2word"]

    with torch.no_grad():
        features = encoder(image_tensor)

        start_token = word2idx["<start>"]
        end_token = word2idx["<end>"]

        caption = [start_token]

        for _ in range(max_len):
            input_seq = torch.tensor(caption).unsqueeze(0).to(device)

            x = decoder_input(features, input_seq)

            # IMPORTANT (match your training!)
            x = x[:, 1:, :]   # ← this was missing

            x = pos_enc(x)

            mask = generate_causal_mask(x.shape[1]).to(device)

            outputs = decoder(x, features.unsqueeze(1), mask)

            next_token = outputs[:, -1, :].argmax(dim=-1).item()
            caption.append(next_token)

            if next_token == end_token:
                break

        # Clean decode
        final_words = []
        for idx in caption[1:]:
            word = idx2word[idx]
            if word == "<end>":
                break
            final_words.append(word)

    return " ".join(final_words)


# ----------------------------
# Beam Search (CORRECT)
# ----------------------------
def beam_search(image_tensor, models, beam_width=3, max_len=30):
    encoder = models["encoder"]
    decoder = models["decoder"]
    decoder_input = models["decoder_input"]
    pos_enc = models["pos_enc"]
    word2idx = models["word2idx"]
    idx2word = models["idx2word"]

    with torch.no_grad():
        features = encoder(image_tensor)

        start_token = word2idx["<start>"]
        end_token = word2idx["<end>"]

        sequences = [[[start_token], 0.0]]

        for _ in range(max_len):
            all_candidates = []

            for seq, score in sequences:

                if seq[-1] == end_token:
                    all_candidates.append([seq, score])
                    continue

                input_seq = torch.tensor(seq).unsqueeze(0).to(device)

                x = decoder_input(features, input_seq)

                # IMPORTANT (same as greedy)
                x = x[:, 1:, :]

                x = pos_enc(x)

                mask = generate_causal_mask(x.shape[1]).to(device)

                outputs = decoder(x, features.unsqueeze(1), mask)

                probs = torch.softmax(outputs[:, -1, :], dim=-1)
                topk_probs, topk_idx = probs.topk(beam_width)

                for i in range(beam_width):
                    candidate = seq + [topk_idx[0][i].item()]

                    # length normalized score
                    candidate_score = score - torch.log(topk_probs[0][i]).item() / len(candidate)

                    all_candidates.append([candidate, candidate_score])

            sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        best_seq = sequences[0][0]

        final_words = []
        for idx in best_seq[1:]:
            word = idx2word[idx]
            if word == "<end>":
                break
            final_words.append(word)

    return " ".join(final_words)


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    models = load_models()

    image_path = "images/eee.jpg"

    print("Greedy:", generate_caption(image_path, models, beam=False))
    print("Beam:", generate_caption(image_path, models, beam=True))