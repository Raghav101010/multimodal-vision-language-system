import gradio as gr
from inference import load_models, generate_caption
import torch
from src.data.tokenizer import InferenceTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

tokenizer = InferenceTokenizer(word2idx, idx2word)

config = {
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "max_len": 100,
    "vocab_size": len(tokenizer.word2idx),
}

models = load_models()

def caption_fn(image):
    return generate_caption(image, models, tokenizer)

demo = gr.Interface(
    fn=caption_fn,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning System"
)

demo.launch()