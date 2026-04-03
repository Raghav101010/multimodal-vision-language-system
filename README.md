# Multimodal Vision-Language System (Image Captioning + VQA)

## Overview

This project is an end-to-end **Multimodal AI System** that combines **Computer Vision and Natural Language Processing** to:

* Generate captions for images
* Answer questions about images (VQA)
* Evaluate model performance using BLEU metrics
* Provide an interactive web interface using Gradio

The system integrates a **custom-trained image captioning model** with a **pretrained VQA transformer model**, forming a unified vision-language pipeline.

---

## Architecture

```
                ┌────────────────────┐
                │     Input Image    │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │  CNN Encoder       │
                │  (ResNet50)        │
                └─────────┬──────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌────────────────────┐            ┌────────────────────────┐
│ Transformer Decoder │            │ VQA Model (ViLT)        │
│ (Caption Generator) │            │ (Pretrained Transformer)│
└─────────┬──────────┘            └──────────┬─────────────┘
          ↓                                  ↓
   Generated Caption                  Answer to Question
```

---

## Model Components

### 1. Image Captioning Model

* **Encoder:** ResNet50 (feature extraction)
* **Decoder:** Transformer (multi-head attention + positional encoding)
* **Decoding Strategies:**

  * Greedy Search
  * Beam Search (improved results)

---

### 2. Visual Question Answering (VQA)

* Model: **ViLT (Vision-and-Language Transformer)**
* Handles natural language queries about the image
* Lightweight and efficient for deployment

---

### 3. Tokenizer

* Custom vocabulary built from dataset
* Special tokens:

  * `<start>`
  * `<end>`
  * `<pad>`
  * `<unk>`

---

## Evaluation

The model is evaluated using **BLEU scores** with multi-caption references (Flickr8k dataset):

```
BLEU-1: 0.76
BLEU-2: 0.61
BLEU-4: 0.39
```

> Multi-caption evaluation significantly improves reliability compared to single-reference BLEU.

---

## Web Interface (Gradio)

The system provides an interactive UI where users can:

* Upload an image
* Generate captions (Greedy / Beam)
* Ask questions about the image
* View model performance metrics

---

## Features

*  End-to-end Image Captioning pipeline
*  Transformer-based decoder
*  Beam Search decoding
*  Multi-caption BLEU evaluation
*  Integrated VQA system
*  Gradio-based UI
*  Modular and scalable codebase

---

##  Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python -m app
```

### 3. Open in browser

```
http://127.0.0.1:7860
```

---

## Project Structure

```
├── app.py
├── inference.py
├── vqa_inference.py
├── evaluation/
│   ├── evaluate.py
│   └── bleu.py
├── src/
├── checkpoints/
├── requirements.txt
├── README.md
```

---

## Future Improvements

* Improve captioning with attention refinement
* Add CIDEr / METEOR metrics
* Fine-tune encoder (ResNet)
* Deploy on HuggingFace Spaces
* Extend to video captioning

---

## Key Highlights

* Built custom Transformer-based captioning model
* Integrated pretrained multimodal transformer (VQA)
* Designed complete inference + evaluation pipeline
* Developed deployable AI application

---

## Author

**Raghwendra Mahato**
AI / ML Engineer (Transitioning from Telecom to AI)
