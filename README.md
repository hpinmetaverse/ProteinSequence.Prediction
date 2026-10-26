# Protein Secondary Structure Prediction (PSSP)

> **JUET Major Project** | Harsh Vardhan Chauhan · Himanshu Sandip Patil · Rudransh Srivastava  
> **Supervisor**: Dr. Ajay Kumar

An end-to-end deep learning pipeline for predicting protein secondary structure in **8-state (Q8)** and **3-state (Q3)** categories from amino acid sequences, using **Bidirectional RNN** (baseline) and **2-layer stacked Bidirectional LSTM** (enhanced model).

---

## Project Structure

```
major 1/
├── docs/
│   ├── PRD.md            # Product Requirements Document
│   ├── ARCHITECTURE.md   # System architecture & model design
│   ├── PHASES.md         # 8-week development plan
│   └── RULES.md          # Engineering & coding standards
├── src/
│   ├── models/
│   │   ├── base_model.py # Abstract base class + Q8→Q3 mapping
│   │   ├── bi_rnn.py     # Baseline Bidirectional RNN
│   │   └── bi_lstm.py    # 2-layer stacked Bidirectional LSTM
│   ├── data/
│   │   ├── dataset.py    # CullPDB / CB513 PyTorch Dataset
│   │   └── collator.py   # Dynamic batch padding collator
│   └── utils/
│       ├── metrics.py    # Q3/Q8 accuracy, F1, confusion matrix
│       └── trainer.py    # Training loop with checkpointing
├── configs/
│   ├── bi_rnn_config.yaml
│   └── bi_lstm_config.yaml
├── scripts/
│   ├── train.py          # CLI training entry point
│   └── evaluate.py       # Test-set evaluation entry point
├── explanation.md        # Model, training, and evaluation rationale
└── tests/
    ├── test_models.py
    ├── test_dataset.py
    └── test_metrics.py
```

---

## Setup

```bash
# Create environment
conda create -n pssp python=3.10
conda activate pssp

# Install dependencies
pip install torch torchvision numpy scikit-learn pyyaml pytest matplotlib seaborn

# Install project as editable package
pip install -e .
```

---

## Data

Download the datasets from the Princeton ICML2014 repository and place them in `data/`:

| File | URL |
|------|-----|
| `cullpdb_pc25_ref2016.npy` | https://www.princeton.edu/~jzthree/datasets/ICML2014/ |
| `cb513.npy` | https://www.princeton.edu/~jzthree/datasets/ICML2014/ |

---

## Training

```bash
# Train 2-layer Bi-LSTM (recommended)
python scripts/train.py --model bi_lstm

# Train Bi-RNN baseline
python scripts/train.py --model bi_rnn

# Custom hyperparameters
python scripts/train.py --model bi_lstm --max_epochs 30 --batch_size 16 --hidden_dim 512

# Resume training from an existing checkpoint
python scripts/train.py --model bi_rnn --max_epochs 50 \
    --resume checkpoints/BiRNN_best.pt
```

Checkpoints are saved to `checkpoints/{model_name}_best.pt`.  
Training history is saved to `logs/{model_name}_history.json`.

## Test-set Evaluation

Evaluate a trained checkpoint on `data/test.csv`:

```bash
python scripts/evaluate.py --model bi_rnn \
    --checkpoint checkpoints/BiRNN_best.pt

python scripts/evaluate.py --model bi_lstm \
    --checkpoint checkpoints/BiLSTM_best.pt
```

The evaluator reports masked per-residue Q3 and Q8 accuracy. The `--model`
arguments must match the model architecture used to create the checkpoint.

## Final Test Metrics

The final checkpoints were evaluated on the independent `data/test.csv` split,
containing 256 proteins:

| Model | Test Q3 Accuracy | Test Q8 Accuracy |
|-------|-----------------:|-----------------:|
| Bi-RNN | 68.01% | 56.78% |
| BiLSTM | 72.35% | 60.07% |

The BiLSTM improves over the Bi-RNN by 4.34 percentage points on Q3 and 3.29
percentage points on Q8. See [explanation.md](explanation.md) for the design
decisions, training rationale, evaluation method, and limitations.

---

## Testing

```bash
# Run all unit tests (no dataset files required)
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v
```

---

## Model Architecture Summary

| Component | Bi-RNN (Baseline) | Bi-LSTM (Enhanced) |
|-----------|-------------------|---------------------|
| Input | (B, L, 21) | (B, L, 21) |
| Embedding | Linear(42→128) + LN + ReLU | Linear(42→128) + LN + ReLU |
| Recurrent | 1× Bi-RNN (hidden=256) | 2× Bi-LSTM (hidden=256) |
| Output dim | 512 | 512 |
| Q8 head | Linear(512→8) | Linear(512→8) |
| Q3 head | None (via mapping) | Linear(512→3) auxiliary |
| Parameters | 204,808 | 2,376,203 |
| Dropout | 0.3 | 0.3 |

---

## Q8 → Q3 Class Mapping (DSSP)

| Q8 Class | Full Name | Q3 Class |
|----------|-----------|----------|
| L | Loop / Irregular | C (Coil) |
| T | Turn | C (Coil) |
| S | Bend | C (Coil) |
| H | Alpha-Helix | H (Helix) |
| I | Pi-Helix | H (Helix) |
| G | 3-10 Helix | H (Helix) |
| E | Beta-Strand | E (Strand) |
| B | Beta-Bridge | E (Strand) |
