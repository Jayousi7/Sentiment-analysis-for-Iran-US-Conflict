# Sentiment Classification for Iran-US Conflict

A deep learning pipeline for classifying sentiment in Arabic tweets related to the **Iran–US geopolitical conflict**. The dataset was **collected manually** from Twitter/X and **manually annotated**, the project compares multiple NLP architectures to identify the best-performing model for this domain-specific task.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameter Search](#hyperparameter-search)
- [Results](#results)
- [License](#license)

---

## Overview

This project tackles **three-class sentiment classification** (Negative, Neutral, Positive) on Arabic-language tweets discussing the Iran–US conflict. We compare:

- **BiLSTM** — A Bidirectional LSTM with learned embeddings
- **AraBERT** — A BERT model pre-trained on Arabic Twitter data ([aubmindlab/bert-base-arabertv02-twitter](https://huggingface.co/aubmindlab/bert-base-arabertv02-twitter))
- **MARBERT** — A BERT model pre-trained on a large-scale Arabic corpus ([UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT))

The pipeline includes automated **grid search** over hyperparameters, **early stopping** with patience, **selective layer freezing** for transformer fine-tuning, **class-imbalance handling** via weighted sampling, and full **per-class metric logging**.

---

## Dataset

> **Note:** The dataset was collected by our team directly from Twitter/X. It is not publicly sourced.

| Split | Samples |
|-------|---------|
| Train | 1,484   |
| Val   |   423   |
| Test  | 214     |
| **Total** | **2,121** |

**Label Distribution (Training Set):**

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Negative  | 975   | 65.7%      |
| Neutral   | 293   | 19.7%      |
| Positive  | 216   | 14.6%      |

Each CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `URL` | Link to the original tweet |
| `Tweet Content` | Raw Arabic text of the tweet |
| `Tweet ID` | Unique Twitter/X post identifier |
| `Sentiment Class` | Label — `Neg`, `Neutral`, or `Pos` |

---

## Models

### 1. Bidirectional LSTM (BiLSTM)
- Embedding dimension: 300
- Hidden dimension: 256
- 2-layer BiLSTM with dropout
- Packed sequence support for variable-length inputs

### 2. AraBERT (Transformer)
- Pre-trained: `aubmindlab/bert-base-arabertv02-twitter`
- Fine-tuned for 3-class sequence classification
- Optimized for Arabic Twitter text
- **Fine-tuning strategy:** Bottom N encoder layers + embeddings optionally frozen; only upper layers and classifier head are updated

### 3. MARBERT (Transformer)
- Pre-trained: `UBC-NLP/MARBERT`
- Fine-tuned for 3-class sequence classification
- Trained on a diverse, large-scale Arabic corpus
- **Fine-tuning strategy:** Same selective layer-freezing approach as AraBERT

---

## Project Structure

```
NLP_PROJ/
├── data/
│   ├── train.csv              # Training split
│   ├── val.csv                # Validation split
│   └── test.csv               # Test split
├── configs.py                 # Hyperparameter grids for grid search
├── Dataset.py                 # TweetDataset class & DataLoader factory
├── models.py                  # BiLSTM and Transformer model definitions
├── train.py                   # Main training entry point (CLI)
├── grid_search.py             # Automated grid search with CSV logging
├── evaluate.py                # Model evaluation & comparison on test set
├── utils.py                   # Metrics calculation & learning curve plots
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd NLP_PROJ

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Usage

### Train a model with grid search

```bash
# Train AraBERT
python train.py --model arabert

# Train MARBERT
python train.py --model marbert

# Train BiLSTM
python train.py --model bilstm
```

#### Optional CLI Arguments:
- `--epochs <N>`: Override the number of training epochs. If omitted, defaults are read from `configs.py` (`epochs` key): **50** for BiLSTM, **20** for Transformers.
- `--dry_run`: Runs a fast, fixed **2-epoch** training run on a small subset (64 samples) to validate the pipeline end-to-end.

### Dry run (pipeline test)

Run a quick 2-epoch test on a tiny data subset to verify everything works:

```bash
python train.py --model arabert --dry_run
```

### Output

After training, results are saved to `<model_name>_results/`:

```
arabert_results/
├── all_configs_results.csv    # All hyperparameter configurations & metrics
└── best_model/
    ├── weights.pt             # Best model weights
    ├── best_results.json      # Best config summary
    └── learning_curves.png    # Loss & metric plots
```

---

## Hyperparameter Search

The grid search explores the following hyperparameters:

**Transformer Models (AraBERT / MARBERT):**

| Parameter | Values | Notes |
|-----------|--------|-------|
| Learning Rate | 2e-5, 3e-5, 5e-5 | Safe fine-tuning range for BERT encoders |
| Batch Size | 16, 32 | |
| Weight Decay | 0.01, 0.005 | |
| Optimizer | Adam, AdamW | |
| Epochs | 20 | |
| Patience | 5 | Early stopping epochs without improvement |
| Freeze Layers | 6, 0 | Number of bottom encoder layers to freeze (0 = full fine-tuning) |

> **Why freeze?** AraBERT/MARBERT have 12 encoder layers. The lower layers encode general Arabic morphology and syntax , already well-learned from pre-training. Freezing them (along with embeddings) prevents catastrophic forgetting on a small dataset (~1.5K samples) and forces adaptation in the upper, more task-sensitive layers.

**BiLSTM:**

| Parameter | Values           | Notes |
|-----------|------------------|-------|
| Learning Rate | 1e-3, 1e-2, 1e-4 | Higher LR appropriate for training from scratch |
| Batch Size | 16, 32           | |
| Weight Decay | 0.01, 0.005      | |
| Dropout | 0.3, 0.5         | |
| Optimizer | Adam, AdamW      | |
| Epochs | 50               | More epochs needed since trained from scratch |
| Patience | 10               | Early stopping epochs without improvement |

---

## Results

Results for each model configuration are logged in `all_configs_results.csv`. The best model (by validation Macro F1) is automatically saved with:

- **Model weights** (`weights.pt`)
- **Metrics summary** (`best_results.json`) — includes per-class precision, recall, and Macro F1
- **Learning curves** (`learning_curves.png`) — training/validation loss and accuracy over epochs

---

## Evaluation

After training one or more models, run the evaluation script to compare their performance on the held-out test set:

```bash
python evaluate.py
```

The script automatically discovers all trained models (by scanning for `*_results/best_model/` directories), evaluates each on the test set, and generates:

```
evaluation_results/
├── model_comparison.csv               # Side-by-side comparison ranked by Macro F1
├── arabert_classification_report.csv   # Per-class precision, recall, F1
├── arabert_confusion_matrix.png        # Confusion matrix heatmap
├── marbert_classification_report.csv
├── marbert_confusion_matrix.png
├── bilstm_classification_report.csv
└── bilstm_confusion_matrix.png
```

Optional arguments:

```bash
python evaluate.py --test_path data/test.csv --batch_size 32 --max_length 128
```

---

## License

This project is for academic and research purposes.
