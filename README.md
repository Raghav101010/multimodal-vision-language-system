Multimodal Vision-Language System (Image Captioning)

Overview
This project implements an end-to-end Image Captioning System using a combination of:
    - CNN Encoder (ResNet)
    - CNN Encoder (ResNet)
    - Transformer Decoder
    - Custom Tokenizer
    - Gradio Web Interface
The system generates natural language descriptions for input images.

Architecture

Image → CNN Encoder (ResNet) → Feature Embedding
       ↓
Transformer Decoder (with Positional Encoding + Masking)
       ↓
Caption Generation (Greedy / Beam Search)

Model Components
1. Encoder
Pretrained ResNet backbone
Extracts visual features
Output projected to embedding dimension
2. Decoder
Transformer-based architecture
Components:
Token Embedding (DecoderInput)
Positional Encoding
Multi-head Attention
Causal Masking
3. Tokenizer
Custom-built vocabulary
Special tokens:
<start>
<end>
<pad>
<unk>

Features Implemented

 Image Caption Generation
 Greedy Decoding
 Beam Search Decoding (Improved results)
 Proper <end> token handling
 Training & Inference Pipeline
 Gradio Web App (Local UI)
 Modular Code Structure

Running the Project Locally
    pip install -r requirements.txt
    python -m inference

    Output:
    Greedy: a woman holding a camera
    Beam: a woman in a red shirt is holding a camera in front of a crowd