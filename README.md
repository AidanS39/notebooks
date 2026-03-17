<div align="center">

![Transformer](./public/logo.svg)

_The Transformer Architecture, built from scratch._

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=fff)](#)
[![Slurm](https://img.shields.io/badge/Slurm-267626?style=for-the-badge&logo=slurm&logoColor=white)](#)
</div>

## Overview

This model is a complete implementation of the state-of-the-art Transformer architecture, built using only basic PyTorch layers and tools. Given a sequence of tokens, this model excels at predicting the next token of the given sequence. This simple task of predicting the next word is the foundation of how modern LLMs are able to generate responses that are not only human-like, but also relevant to the prompt it was given.

## Requirements
 
- Python 3.8+
- CUDA-capable GPU (strongly recommended; CPU fallback available)
- The following Python packages:
 
```
torch
numpy
pandas
datasets
tiktoken
torchvision
```
 
## Installation
 
1. **Clone the repository:**
 
```bash
git clone https://github.com/AidanS39/notebooks.git
cd notebooks/Transformer
```
 
2. **Create and activate a virtual environment (recommended):**
 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
 
3. **Install dependencies:**
 
```bash
pip install torch torchvision datasets tiktoken numpy pandas
```
 
4. **Create the models and output directories** (if they don't already exist):
 
```bash
mkdir -p models output
```
 
## Usage
 
### Training
 
Run `train.py` to download the TinyStories dataset, tokenize it, initialize the model, and begin training:
 
```bash
python3 train.py
```
 
You can override any hyperparameter via command-line flags:
 
```bash
python3 train.py -m 512 -h 4 -l 4 -u 1024 -e 3
```
 
| Flag | Argument | Default | Description |
|---|---|---|---|
| `-m` | `--d_model` | `1024` | Model dimension |
| `-h` | `--n_heads` | `8` | Number of attention heads |
| `-l` | `--n_layers` | `8` | Number of Transformer layers |
| `-u` | `--d_up` | `2048` | MLP up-projection dimension |
| `-e` | `--n_epochs` | `5` | Number of training epochs |
 
Checkpoints are saved to `./models/` automatically during training (see [Checkpointing](#checkpointing)).
 
### Generation

Run `generate.py` to launch an interactive story generation session:
```bash
python3 generate.py
```

On startup, the program scans the `./models/` directory and presents a numbered list of available checkpoints to choose from. After selecting a model, it loads the checkpoint and enters a prompt loop — type the beginning of a story and press `ENTER` to generate a continuation. To exit, press `ENTER` with an empty prompt or type `exit`.

**CLI flags:**

| Flag | Argument | Default | Description |
|---|---|---|---|
| `-f` | `--models_path` | `./models` | Path to the folder containing model checkpoints |
| | `--no-delay` | off | Disable the typewriter-style print delay |

> Note: The first response will be slower than subsequent ones, as the model fully loads during the initial forward pass. Later responses will be significantly faster.

Generation uses **top-k sampling** (`k=3`) with **temperature scaling** (`temp=1.5`) and stops after 1,000 tokens or upon encountering an end-of-sequence token.
 
### SLURM Job Submission
 
For training on an HPC cluster, use the provided SLURM batch script:
 
```bash
sbatch train_job.sh
```
 
The script requests 1 GPU node (L40S GPU) with 16 CPU cores and a 5-day time limit. Model hyperparameters can be overridden at submission time:
 
```bash
sbatch train_job.sh -m 512 -h 4 -l 4 -u 1024 -e 3
```
 
Job output is written to `./output/job.<jobid>.out`. Email notifications on job start, end, and failure are configured within the script — update the `--mail-user` field before submitting.
 
## Configuration
 
The `TransformerConfig` class in `model.py` bundles all model hyperparameters and is saved as part of every checkpoint, ensuring that the correct architecture is always reconstructed at inference time without any manual configuration.
 
Checkpoint filenames encode the model's hyperparameters for easy identification:
 
```
models/model_{d_model}_{n_heads}_{n_layers}_{d_up}.pt
```
 
For example, a model with `d_model=1024`, `n_heads=8`, `n_layers=8`, `d_up=2048` is saved as:
 
```
models/model_1024_8_8_2048.pt
```
 
## Checkpointing
 
Training supports full checkpoint saving and resumption. Checkpoints are saved:
 
- Every `64 * accum_steps` batches during training
- At the end of every epoch
- At the completion of all training
 
If a checkpoint already exists at the expected path when `train.py` is launched, training automatically resumes from where it left off — including the optimizer state, RNG state, epoch, and batch index. This makes it safe to resume after a SLURM job timeout or interruption.
 
The `CheckpointRandomSampler` ensures that already-seen batches within a partial epoch are skipped on resume.


>⚠️ **Do not interrupt the process while "Saving model checkpoint... DO NOT EXIT" is printed**, as this may corrupt the checkpoint file.
 
## Training Details
 
- **Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — split 80% train / 20% test, with a small 0.1% validation subset held out for periodic loss evaluation during training.
- **Tokenizer:** `cl100k_base` from `tiktoken` (OpenAI's tokenizer, ~100k vocabulary). A custom `<EOS>` token is appended to every sequence; a padding token is used to collate variable-length sequences in each batch.
- **Loss:** Cross-entropy over next-token predictions, with the padding token ignored.
- **Precision:** Automatic mixed precision using `bfloat16` via `torch.autocast`.
- **Compilation:** `torch.compile` is applied to the model before training for improved throughput.
- **Gradient accumulation:** Gradients are accumulated over `accum_steps=4` batches before each optimizer step, effectively increasing the batch size without increasing VRAM usage.
- **Validation:** Model validation loss is computed every `64 * accum_steps` batches during training.