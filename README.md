# RNNoise Fine-Tuning — On-Device Noise Cancellation

This repository contains the training pipeline used to fine-tune the [RNNoise](https://github.com/xiph/rnnoise) model on a custom dataset for improved, real-time, on-device noise suppression.

## Overview

RNNoise is a lightweight recurrent neural network for real-time noise suppression, well suited for on-device deployment. This project fine-tunes RNNoise on a custom mix of clean speech and real-world noise to improve denoising quality while preserving its small footprint and real-time performance characteristics.

The pipeline covers:
- Environment setup (system + Python dependencies, GPU verification)
- Building RNNoise from source (C feature extractor + inference demo)
- Curating and preparing clean speech and noise datasets
- Generating training features by mixing speech and noise
- Training the model (PyTorch implementation) with checkpointing
- Exporting trained weights back into RNNoise's native C format
- Rebuilding the C library with the fine-tuned weights
- Testing the fine-tuned model on real noisy audio samples
- Packaging the final model artifacts for distribution

## Pipeline Steps

### 1. Environment Setup
- Installs system dependencies: `autoconf`, `automake`, `libtool`, `gcc`, `make`, `git`, `sox`, `ffmpeg`, `wget`, `unzip`
- Installs Python dependencies: `pesq`, `pystoi`, `soundfile`, `librosa`, `tqdm`, `torch`
- Verifies GPU availability (TensorFlow / CUDA)

### 2. Build RNNoise from Source
- Clones the official [xiph/rnnoise](https://github.com/xiph/rnnoise) repository
- Builds it via `autogen.sh`, `configure`, and `make`
- Compiles the feature extraction tool (`dump_features`)

### 3. Dataset Preparation
- **Clean speech:** Streamed from standard datasets (`SPRINGLab/IndicTTS-Hindi`, `SPRINGLab/IndicTTS-English` via Hugging Face `datasets`), resampled to 48kHz mono, and saved as raw PCM16 (`speech.raw`)
- **Noise:** Curated custom noise audio files (converted from MP3 to 48kHz mono raw PCM via `ffmpeg`) saved as `noise.raw`

### 4. Feature Extraction
- Uses RNNoise's `dump_features` binary to mix speech and noise samples and generate training features (`features.f32`)
- Multiple feature batches are generated and merged into a single dataset

### 5. Data Formatting
- Raw float features are reshaped into RNNoise's expected frame format (87 columns per row)
- Converted into HDF5 format (`training.h5`) for training

### 6. Model Training
- Training is performed using RNNoise's PyTorch implementation (`torch/rnnoise/train_rnnoise.py`)
- Key hyperparameters:
  - GRU size: `384`
  - Conditioning size: `128`
  - Batch size: `32`
  - Sequence length: `500`
  - Epochs: `100`
  - Learning rate: `1e-3`
  - Sparsification enabled (`--sparse`)
- Checkpoints are saved per epoch, with training loss tracked and plotted to identify the best-performing checkpoint

### 7. Weight Export
- The trained PyTorch checkpoint (`.pth`) is converted into RNNoise's native C format using `dump_rnnoise_weights.py`
- Produces `rnnoise_data.c` and `rnnoise_data.h`

### 8. Rebuild with Fine-Tuned Weights
- Replaces the default `rnnoise_data.c` / `rnnoise_data.h` in the RNNoise source with the exported fine-tuned weights
- Rebuilds the library (`make clean && make`) to produce an updated `rnnoise_demo` binary

### 9. Evaluation
- Real noisy audio samples are converted to 16kHz mono raw PCM via `ffmpeg`
- Passed through the rebuilt `rnnoise_demo` binary for denoising
- Noisy vs. denoised outputs are compared by listening (via `IPython.display.Audio`) across multiple checkpoints to select the best-performing model

### 10. Packaging
- Final model artifacts (checkpoint `.pth`, exported `rnnoise_data.c`/`.h`) are zipped for distribution and persistence (Kaggle dataset / Google Drive)

## Requirements

- Python 3.x
- PyTorch (with CUDA support recommended)
- TensorFlow (for GPU verification)
- `librosa`, `soundfile`, `pesq`, `pystoi`, `tqdm`, `h5py`, `datasets`
- System tools: `ffmpeg`, `sox`, `autoconf`, `automake`, `libtool`, `gcc`, `make`, `git`
- GPU recommended for training (CUDA-enabled)

## Project Structure

```
rnnoise/                       # Cloned RNNoise source repository
├── src/                       # C source, includes rnnoise_data.c/.h (model weights)
├── examples/                  # rnnoise_demo binary for inference/testing
├── training/                  # Legacy Keras training scripts + bin2hdf5.py
└── torch/rnnoise/             # PyTorch training implementation
    ├── train_rnnoise.py       # Main training script
    ├── rnnoise.py             # Model definition
    └── dump_rnnoise_weights.py # Exports trained weights to C format

speech.raw                     # Clean speech dataset (PCM16, 48kHz, mono)
noise.raw                      # Curated noise dataset (PCM16, 48kHz, mono)
features.f32 / features_merged.f32  # Extracted training features
training.h5                    # Formatted training data (HDF5)
output/checkpoints/             # Training checkpoints (.pth)
exported_weights/               # Exported rnnoise_data.c / .h
rnnoise_trained_model.zip       # Final packaged model
```

## Usage

1. Set up the environment and build RNNoise from source.
2. Prepare `speech.raw` and `noise.raw` datasets.
3. Run `dump_features` to generate training features.
4. Convert features to HDF5 format.
5. Train the model using `train_rnnoise.py` with the desired hyperparameters.
6. Export the best checkpoint to C format using `dump_rnnoise_weights.py`.
7. Replace `rnnoise_data.c` / `.h` and rebuild RNNoise.
8. Test the fine-tuned model using `rnnoise_demo` on real noisy audio samples.

## Notes

- Multiple checkpoints were exported and evaluated (e.g., epoch 15, 75, 99, 100) to identify the best trade-off between denoising quality and training convergence.
- The fine-tuned model maintains RNNoise's lightweight, real-time, on-device design while improving noise suppression performance on real-world noise conditions.
