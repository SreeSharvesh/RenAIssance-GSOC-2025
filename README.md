# Optical Character Recognition (OCR) with TrOCR

## Overview
This repository contains the implementation of **Specific Test II: Optical Character Recognition (OCR)**, a task focused on recognizing and extracting text from preprocessed line segment images of scanned documents. The objective is to accurately transcribe the main text content while disregarding embellishments such as borders, noise, or irrelevant markings. This approach leverages **TrOCR (Transformer-based Optical Character Recognition) model**, a state-of-the-art transformer architecture that combines a **Vision Transformer (ViT) encoder** and a **transformer decoder**.

## Task Overview
- **Task**: Develop a model based on convolutional-recurrent, transformer, or self-supervised architectures to optically recognize text from various data sources, focusing on the main text while ignoring embellishments.
- **Approach**: Fine-tune the **pretrained TrOCR model (VisionEncoderDecoderModel)** from Hugging Face on a dataset of line segment images and their corresponding ground truth text.
- **Input**: Line segment images generated from a prior layout organization step (Task I).
- **Output**: Transcribed text for each line segment.

## Theoretical Background

### TrOCR Model Architecture
TrOCR is a transformer-based OCR model that eliminates the need for traditional convolutional-recurrent architectures (e.g., CNN + LSTM). It consists of two main components:

#### Vision Transformer (ViT) Encoder
- **Purpose**: Extracts visual features from input images.
- **Theory**:
  - The ViT divides the input image into fixed-size patches (e.g., 16x16 pixels), flattens them into a sequence of vectors, and adds positional embeddings to retain spatial information.
  - These patch embeddings are processed through multiple transformer layers, which use self-attention mechanisms to capture global dependencies across the image.
  - The output is a sequence of feature vectors representing the visual content of the image.
- **Advantages**: Unlike CNNs, ViT captures long-range dependencies without locality bias, making it robust for varied text layouts.

#### Transformer Decoder
- **Purpose**: Generates the text sequence from the encoded visual features.
- **Theory**:
  - The decoder is a standard transformer architecture with masked self-attention, allowing it to predict the next token in the sequence autoregressively.
  - It uses cross-attention to align visual features from the encoder with the text sequence being generated.
  - The model is trained with a causal language modeling objective, predicting each character or token based on prior ones.
- **Advantages**: The transformer decoder excels at sequence modeling, leveraging its attention mechanism to handle variable-length text outputs.

### End-to-End Design
- The encoder and decoder are jointly trained, enabling the model to learn a direct mapping from image pixels to text without intermediate bounding box or segmentation steps.
- Pretraining on large datasets (e.g., synthetic text images) provides a strong starting point, which is fine-tuned for specific tasks.

### Why TrOCR?
- **Transformer Superiority**: Replaces recurrent layers (e.g., LSTM) with transformers, offering parallelization, scalability, and better performance on sequential data.
- **Pretraining**: Leverages pretrained weights from Hugging Face, reducing training time and data requirements.
- **Flexibility**: Handles diverse text styles, fonts, and layouts without explicit preprocessing for each case.

## Approach and Strategy

### Process Overview
#### 1. Data Preparation
- **Input**:
  - Line segment images which can be obtained using teh CRAFT algorithm.
  - Ground truth text from a CSV file (e.g., `datas.csv` with columns `image_path` and `text`) which contains the mapping of ground truth and the correspponding line segment image.
- **Preprocessing**:
  - **Resizing**: Images are resized to a uniform size (256x64 pixels) to match TrOCR’s expected input dimensions.
  - **Augmentation**: `albumentations` is used for data augmentation (e.g., random cropping, rotation) to improve model robustness.
  - **Processor**: `TrOCRProcessor` tokenizes ground truth text and prepares pixel values for the model.
- **Dataset Creation**:
  - A custom PyTorch `Dataset` class is implemented to load and preprocess image-text pairs.
  - `DataLoader` with batching (e.g., `batch_size=2`) and padding (via `pad_sequence`) ensures efficient training.

#### 2. Model Setup
- **Model**:
  - `VisionEncoderDecoderModel.from_pretrained("qantev/trocr-large-spanish")` loads the pretrained TrOCR model optimized for handwritten text (adaptable to printed text).
  - **Configuration details**:
    - Encoder (ViT): 24 layers, 1024 hidden size, 16 attention heads, 384x384 image input.
    - Decoder: 12 layers, 1024 hidden size, 16 attention heads, max position embeddings of 1024.
- **Processor**:
  - `TrOCRProcessor.from_pretrained("qantev/trocr-large-spanish")` handles image normalization and text tokenization.
- **Device**:
  - Set to CPU (`torch.set_default_device("cpu")`) due to `CUDA_VISIBLE_DEVICES=""`, but can be switched to GPU for faster training. (I did not have a GPU)

#### 3. Training
- **Optimizer**: AdamW with default parameters for adaptive learning rate optimization.
- **Scheduler**: ReduceLROnPlateau reduces the learning rate when validation loss plateaus, improving convergence.
- **Loss Function**: Cross-entropy loss is implicitly used via the model’s causal language modeling objective.
- **Training Loop**:
  - Implemented using `Trainer` from Hugging Face or a custom PyTorch loop.
  - Batches of image-text pairs are fed to the model, with gradients computed and weights updated.

#### 4. Evaluation
- **Function**: `evaluate_model` computes predictions and metrics.
- **Steps**:
  - Load line images and ground truth text.
  - Generate predictions using the fine-tuned model.
  - Compute CER and WER using the `jiwer` library.
- **Output**:
  - Average CER and WER scores across all predictions.
  - Individual predictions printed for inspection.

## Evaluation Metrics
### Character Error Rate (CER)
- **Definition**: The ratio of character-level errors (insertions, deletions, substitutions) to the total number of characters in the ground truth.
- **Formula**:
  
  CER = (S + D + I) / N
  
  where:
  - S = Substitutions
  - D = Deletions
  - I = Insertions
  - N = Total characters
- **Purpose**: Measures fine-grained accuracy, critical for OCR where character precision matters.

### Word Error Rate (WER)
- **Definition**: The ratio of word-level errors (insertions, deletions, substitutions) to the total number of words in the ground truth.
- **Formula**:
  WER = (S_w + D_w + I_w) / N_w
  
  where:
  - S_w, D_w, I_w = Word-level errors
  - N_w = Total words
- **Purpose**: Evaluates overall text readability, useful for assessing sentence-level accuracy.

### Average CER/WER
- Computed across all predictions to provide a summary of model performance.


### Training Performance Table

| Epoch | Training Loss | CER  | WER  |
|-------|--------------|------|------|
| 20 |	0.056600 |	0.060328 |	0.006397 |	0.021073 |
| 40 |	0.051600 |	0.064336 |	0.008884 |	0.030651 |
| 60 |	0.048300 |	0.051742 |	0.010661 |	0.022989 |
| 80 |  0.054900 |	0.072586 |	0.010661 |	0.026820 |


### Comments on Specific Task 3:
I tried Speific task 3. I designed a simple GAN based generative model to generate Synthetic renaissance text

