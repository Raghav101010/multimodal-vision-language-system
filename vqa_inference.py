processor = None
model = None

def load_vqa():
    global processor, model
    if processor is None:
        from transformers import ViltProcessor, ViltForQuestionAnswering
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

def answer_question(image, question):
    load_vqa()

    encoding = processor(image, question, return_tensors="pt").to(model.device)
    outputs = model(**encoding)

    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]