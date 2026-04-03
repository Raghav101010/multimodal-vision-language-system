import gradio as gr
from inference import load_models, generate_caption
from vqa_inference import answer_question
import torch
import json
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

with open("bleu_score.json", "r") as f:
    bleu_scores = json.load(f)

def multimodal_fn(image, question, Decoding):
    if Decoding == "Greedy":
        caption = generate_caption(image, models, beam=False)
    else:
        caption = generate_caption(image, models, beam=True)

    answer = answer_question(image, question)
    bleu_text = f"BLEU-1: {bleu_scores['BLEU-1']:.3f}, BLEU-4: {bleu_scores['BLEU-4']:.3f}"

    return caption, answer, bleu_text
    
demo = gr.Interface(
    fn=multimodal_fn,
    inputs=[gr.Image(type="pil"),
            gr.Textbox(label="Ask a question about the image"),
            gr.Radio(["Greedy", "Beam"], value="Beam")],
    outputs=[
        gr.Textbox(label="Generated Caption"),
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Model Performance (BLEU)")
    ],
    title="Multimodal Vision-Language System (Image Captioning)"
)

demo.launch()