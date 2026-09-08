# System Architecture
## Protein Secondary Structure Prediction (PSSP) Pipeline

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PSSP End-to-End Pipeline                        │
│                                                                         │
│  ┌────────────┐    ┌───────────────┐    ┌────────────┐    ┌──────────┐ │
│  │  Raw Input │───>│  Preprocessing│───>│  DL Model  │───>│  Output  │ │
│  │  (Sequence)│    │  & Features   │    │ (Bi-RNN /  │    │ (Q8/Q3   │ │
│  └────────────┘    └───────────────┘    │  Bi-LSTM)  │    │  labels) │ │
│                                         └────────────┘    └──────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Layer

### 2.1 Datasets

| Dataset | Proteins | Purpose | Format |
|---------|----------|---------|--------|
| CullPDB (filtered) | ~5,534 | Training + Validation | NumPy .npy, shape (N, 700, 57) |
| CB513 | 513 | Independent Test | NumPy .npy, shape (513, 700, 57) |

Both datasets follow the CullPDB column layout:
- Columns 0–21: One-hot encoded amino acids (21 types + NoSeq)
- Columns 22–30: Q8 secondary structure labels (8 classes)
- Columns 31–33: Q3 secondary structure labels (3 classes)
- Columns 34–56: PSSM profile (21 values per residue)

### 2.2 Q8 → Q3 Label Mapping

```
Q8 Labels → Q3 Labels
─────────────────────
L (Loop/Coil)      → C (Coil)
T (Turn)           → C (Coil)
S (Bend)           → C (Coil)
H (Alpha-Helix)    → H (Helix)
I (Pi-Helix)       → H (Helix)
G (3-10 Helix)     → H (Helix)
E (Beta-Strand)    → E (Strand)
B (Beta-Bridge)    → E (Strand)
```

### 2.3 Input Feature Vector

Per residue, the model receives a **42-dimensional vector**:
- 21 dims: One-hot amino acid encoding
- 21 dims: PSSM evolutionary profile

---

## 3. Model Layer

### 3.1 Bi-RNN Baseline

```
Input: (Batch, SeqLen, 42)
         │
         ▼
┌──────────────────────────┐
│   Embedding Projection   │  Linear(42 → 128)
│   + LayerNorm + Dropout  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Bidirectional RNN      │  hidden=256 per direction (512 total)
│   (1 layer, tanh)        │  Processes N→C and C→N simultaneously
└──────────┬───────────────┘
           │  (Batch, SeqLen, 512)
           ▼
┌──────────────────────────┐
│   Dropout (p=0.3)        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Output Projection      │  Linear(512 → 8) for Q8
│   (Q8 classifier)        │  ArgMax → Q3 mapping
└──────────────────────────┘
Output: (Batch, SeqLen, 8)
```

**Parameter count**: ~3.5M

### 3.2 Bi-LSTM (Primary Model)

```
Input: (Batch, SeqLen, 42)
         │
         ▼
┌──────────────────────────┐
│   Embedding Projection   │  Linear(42 → 128)
│   + LayerNorm + Dropout  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Bidirectional LSTM      │  Layer 1: hidden=256, bidirectional
│       Layer 1            │  Output: (Batch, SeqLen, 512)
│  [forget, input, output, │
│   cell gates × 2 dirs]   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│     Dropout (p=0.3)      │  Between stacked LSTM layers
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Bidirectional LSTM      │  Layer 2: hidden=256, bidirectional
│       Layer 2            │  Output: (Batch, SeqLen, 512)
│  [Long-range motifs via  │
│   cell state memory]     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Dropout (p=0.3)        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Output Projection      │  Linear(512 → 8) Q8 head
│   + Q3 Projection        │  Linear(512 → 3) Q3 head (auxiliary)
└──────────────────────────┘
Output: (Batch, SeqLen, 8)  +  (Batch, SeqLen, 3)
```

**Parameter count**: ~8.5M

**Why LSTM over RNN?**
- **Gated memory**: The cell state `c_t` persists across hundreds of residues, enabling capture of long-range structural motifs (e.g., alpha-helix repeats spanning 20+ residues).
- **Vanishing gradient mitigation**: Forget/input gates maintain gradients through long sequences.
- **Stacked layers**: Two stacked Bi-LSTM layers allow hierarchical feature abstraction (local contacts → global motifs).

---

## 4. Training Infrastructure

### 4.1 Loss Function

```
Total Loss = CrossEntropy(Q8_pred, Q8_true) [+ optional λ * CE(Q3_pred, Q3_true)]
```

- Ignores padded positions (padding_idx masking via loss ignore_index)
- Class weights optionally applied for imbalanced Q8 classes

### 4.2 Optimization

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| LR Scheduler | ReduceLROnPlateau (patience=3) |
| Gradient Clip | 1.0 (norm) |
| Batch Size | 32 |
| Max Epochs | 50 |
| Dropout | 0.3 |
| Weight Decay | 1e-5 |

### 4.3 Custom Batch Collator

Dynamic padding collates each batch to the **maximum sequence length within that batch** (not global max), dramatically reducing wasted computation on short sequences.

```
collate_fn(batch):
  max_len = max(len(seq) for seq in batch)
  pad each seq to max_len
  return padded_inputs, padded_targets, lengths_mask
```

### 4.4 Checkpointing Strategy

```
for each epoch:
  val_q3 = evaluate(model, val_loader)
  if val_q3 > best_val_q3:
    best_val_q3 = val_q3
    torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pt")
```

Early stopping after `patience=10` epochs with no improvement.

---

## 5. Evaluation Layer

### 5.1 Metrics

| Metric | Description |
|--------|-------------|
| Q3 Accuracy | % residues with correct 3-state prediction |
| Q8 Accuracy | % residues with correct 8-state prediction |
| Per-class F1 | Weighted F1 per structural class |
| Confusion Matrix | Q3 and Q8 confusion on CB513 test set |
| Training Curves | Loss + accuracy per epoch |

### 5.2 Evaluation Masks

Padded positions (NoSeq tokens) are **excluded** from all accuracy calculations using the `lengths` mask. This ensures metrics reflect true predictive performance, not trivial padding predictions.

---

## 6. Directory Structure

```
major 1/
├── docs/
│   ├── PRD.md                  # Product Requirements Document
│   ├── ARCHITECTURE.md         # This file
│   ├── PHASES.md               # Weekly project phases
│   └── RULES.md                # Engineering & coding rules
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bi_rnn.py           # Baseline Bidirectional RNN model
│   │   ├── bi_lstm.py          # 2-layer stacked Bi-LSTM model
│   │   └── base_model.py       # Abstract base class for all models
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # CullPDB / CB513 PyTorch Dataset
│   │   └── collator.py         # Dynamic padding collator
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Q3/Q8 accuracy, F1, confusion matrix
│       ├── trainer.py          # Training loop with checkpointing
│       └── visualization.py    # Plotting training curves
│
├── configs/
│   ├── bi_rnn_config.yaml      # Hyperparameters for Bi-RNN
│   └── bi_lstm_config.yaml     # Hyperparameters for Bi-LSTM
│
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── evaluate.py             # Evaluation & test-set inference
│   └── predict.py              # Single-sequence inference script
│
├── tests/
│   ├── test_models.py          # Unit tests for model forward passes
│   ├── test_dataset.py         # Unit tests for data loading
│   └── test_metrics.py         # Unit tests for metric calculations
│
├── data/                       # (gitignored) Raw .npy dataset files
│   ├── cullpdb_pc25_ref2016.npy
│   └── cb513.npy
│
├── checkpoints/                # (gitignored) Saved model weights
├── notebooks/                  # Exploratory analysis notebooks
├── RULES.md                    # Project rules (root-level quick access)
└── README.md
```

---

## 7. Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch >= 2.0 |
| Data Processing | NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Configuration | PyYAML |
| Testing | pytest |
| Version Control | Git |
| Environment | Conda / pip venv |
| (Future) Web Backend | Flask / FastAPI |
| (Future) Web Frontend | React / Vanilla JS |

---

## 8. Data Flow Diagram

```
CullPDB .npy
     │
     ▼
ProteinDataset.__getitem__()
  - Extract one-hot (cols 0-21)
  - Extract PSSM (cols 35-56)
  - Concatenate → (SeqLen, 42) features
  - Extract Q8 labels (cols 22-30) → (SeqLen,)
  - Extract Q3 labels (cols 31-33) → (SeqLen,)
  - Trim trailing NoSeq residues
     │
     ▼
DynamicPaddingCollator(batch)
  - Find max_len in batch
  - Pad features to (max_len, 42)
  - Pad labels to (max_len,)
  - Build lengths mask (bool tensor)
     │
     ▼
BiLSTM.forward(x, lengths)
  - Project: (B, L, 42) → (B, L, 128)
  - LSTM Layer 1: (B, L, 128) → (B, L, 512)
  - Dropout
  - LSTM Layer 2: (B, L, 512) → (B, L, 512)
  - Q8 head: Linear(512, 8) → logits (B, L, 8)
  - Q3 head: Linear(512, 3) → logits (B, L, 3)
     │
     ▼
CrossEntropyLoss (masked, ignoring padding)
     │
     ▼
Adam.step() + LR scheduler
     │
     ▼
Checkpoint if val_q3 improves
```
