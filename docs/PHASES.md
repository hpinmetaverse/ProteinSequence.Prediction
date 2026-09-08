# Project Phases
## Protein Secondary Structure Prediction (PSSP)
### 8-Week Development Schedule

---

> **Reference**: Based on the weekly project flow outlined in the course slides.
> Each phase builds directly on the previous one, following a classical ML project lifecycle.

---

## Phase Overview

```
Week 1          Week 2          Week 3          Week 4
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Problem  │──>│  Data    │──>│ Baseline │──>│Enhanced  │
│ Framing  │   │ Pipeline │   │  Model   │   │  Model   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘

Week 5          Week 6          Week 7          Week 8
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Analysis │──>│  Web App │──>│Integration│──>│ Final    │
│& Tuning  │   │(Backend) │   │& Testing  │   │ Report   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## Phase 1 — Problem Framing & Project Setup
**Week 1 | Status: COMPLETE**

### Objectives
- [x] Define the research problem: Q8 and Q3 protein secondary structure prediction
- [x] Survey related work (PSIPRED, NetSurfP, DeepCNF, etc.)
- [x] Select datasets: CullPDB (train/val) + CB513 (test)
- [x] Agree on model architectures: Bi-RNN (baseline) vs. 2-layer Bi-LSTM (enhanced)
- [x] Set up Git repository and development environment
- [x] Write PRD and Architecture documentation
- [x] Establish coding standards (see `RULES.md`)

### Deliverables
- `docs/PRD.md`
- `docs/ARCHITECTURE.md`
- `docs/RULES.md`
- `docs/PHASES.md` (this file)
- Initialized project directory structure
- Conda/pip environment with PyTorch, NumPy, scikit-learn

### Key Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use CullPDB (< 25% identity) | Avoids sequence homology bias in train/test split |
| 42-dim input features | 21 one-hot amino acids + 21 PSSM profile columns |
| Q8 as primary prediction target | Q3 derived by class mapping; Q8 is harder and more informative |
| Dynamic batch padding | Minimizes wasted compute on padded positions |

---

## Phase 2 — Data Pipeline
**Week 2 | Status: IN PROGRESS**

### Objectives
- [ ] Download and verify CullPDB and CB513 dataset files
- [ ] Implement `ProteinDataset` (PyTorch Dataset class)
- [ ] Implement `DynamicPaddingCollator` (custom collate_fn)
- [ ] Explore and document dataset statistics (sequence length dist., class frequencies)
- [ ] Write unit tests for data loading and collation
- [ ] Visualize class imbalance (Q8 and Q3 label distributions)

### Key Implementation Tasks

```
src/data/
├── dataset.py    # ProteinDataset, feature extraction, label parsing
└── collator.py   # DynamicPaddingCollator with masking
```

### Dataset Statistics to Compute
- Sequence length distribution (min, max, mean, median, percentiles)
- Q8 class frequencies and imbalance ratio
- Q3 class frequencies
- Number of NoSeq (padding) tokens per protein

### Acceptance Criteria
- [ ] DataLoader iterates without errors at batch_size=32
- [ ] Feature tensors: shape `(B, L, 42)`, dtype `float32`
- [ ] Label tensors: shape `(B, L)`, dtype `int64`
- [ ] Masks: shape `(B, L)`, dtype `bool` (True = real residue)
- [ ] Unit tests passing in `tests/test_dataset.py`

---

## Phase 3 — Baseline Bi-RNN Model
**Week 3 | Status: IN PROGRESS**

### Objectives
- [ ] Implement `BiRNN` model class with configurable hyperparameters
- [ ] Implement training loop with masked CrossEntropy loss
- [ ] Implement validation loop with Q3 and Q8 accuracy computation
- [ ] Implement checkpointing (save best validation Q3 model)
- [ ] Train Bi-RNN on CullPDB for 30-50 epochs
- [ ] Evaluate on CB513 test set
- [ ] Log and plot training curves

### Model Architecture
- Input projection: Linear(42 → 128) + LayerNorm + Dropout(0.3)
- Bidirectional RNN: 1 layer, hidden=256 (×2 directions = 512)
- Output head: Linear(512 → 8) for Q8 classification

### Hyperparameters (Baseline)
| Parameter | Value |
|-----------|-------|
| Embedding dim | 128 |
| Hidden units | 256 |
| Directions | 2 (bidirectional) |
| Layers | 1 |
| Dropout | 0.3 |
| Optimizer | Adam (lr=1e-3) |
| Batch size | 32 |
| Max epochs | 50 |

### Acceptance Criteria
- [ ] Model forward pass: input `(B, L, 42)` → output `(B, L, 8)`
- [ ] Masked loss ignores padded positions
- [ ] Q3 accuracy tracked per epoch
- [ ] Best checkpoint saved automatically
- [ ] Unit tests in `tests/test_models.py`

---

## Phase 4 — Enhanced Bi-LSTM Model
**Week 4 | Status: IN PROGRESS**

### Objectives
- [ ] Implement `BiLSTM` model class with 2 stacked bidirectional LSTM layers
- [ ] Add auxiliary Q3 prediction head (optional regularization)
- [ ] Train Bi-LSTM on CullPDB using same data pipeline as Bi-RNN
- [ ] Apply gradient clipping (max_norm=1.0) to handle exploding gradients
- [ ] Evaluate on CB513 test set
- [ ] Compare Bi-LSTM vs Bi-RNN performance

### Model Architecture
- Input projection: Linear(42 → 128) + LayerNorm + Dropout(0.3)
- Bi-LSTM Layer 1: hidden=256, bidirectional → output 512
- Dropout(0.3) between layers
- Bi-LSTM Layer 2: hidden=256, bidirectional → output 512
- Q8 head: Linear(512 → 8)
- Q3 head: Linear(512 → 3) [auxiliary]

### Why LSTM Over RNN?
- Cell state preserves long-range information across 100+ residue spans
- Forget gate learns to discard irrelevant local context
- Stacked layers learn hierarchical structural representations
- Gradient flow improved through gating mechanism

### Acceptance Criteria
- [ ] Bi-LSTM Q8 accuracy > Bi-RNN Q8 accuracy on CB513
- [ ] Bi-LSTM Q3 accuracy > Bi-RNN Q3 accuracy on CB513
- [ ] No NaN losses during training (gradient clipping active)
- [ ] Best checkpoint saved automatically

---

## Phase 5 — Analysis, Ablation & Hyperparameter Tuning
**Week 5 | Status: TODO**

### Objectives
- [ ] Generate confusion matrices for both models on CB513
- [ ] Compute per-class precision, recall, F1 for Q8 and Q3
- [ ] Ablation: effect of bidirectionality vs. unidirectional LSTM
- [ ] Ablation: 1-layer vs. 2-layer LSTM
- [ ] Ablation: PSSM features vs. one-hot only
- [ ] Hyperparameter sweep: dropout rate, hidden size, learning rate
- [ ] Document findings in `notebooks/05_analysis.ipynb`

### Analysis Questions to Answer
1. Which Q8 class is hardest to predict? (Expected: B - Beta-bridge, rare class)
2. Does PSSM dramatically improve accuracy over one-hot alone?
3. How does training accuracy vs validation accuracy diverge? (Overfitting analysis)
4. What is the minimum sequence length for reliable prediction?

---

## Phase 6 — Web Application Backend
**Week 6 | Status: TODO**

### Objectives
- [ ] Set up Flask/FastAPI server
- [ ] Create `/predict` endpoint accepting FASTA sequence
- [ ] Load best Bi-LSTM checkpoint for inference
- [ ] Return predictions as JSON: `{residue, aa, q8, q3, confidence}`
- [ ] Add health check endpoint `/health`
- [ ] Write API documentation

### API Design
```
POST /api/predict
Content-Type: application/json

{
  "sequence": "ACDEFGHIKLMNPQRSTVWY..."
}

Response 200:
{
  "q8": "HHHHHEEETTTSSS...",
  "q3": "HHHHHEEECCCCCC...",
  "residues": [
    {"position": 0, "aa": "A", "q8": "H", "q3": "H", "confidence": 0.95},
    ...
  ],
  "length": 150,
  "model": "BiLSTM",
  "version": "1.0"
}
```

---

## Phase 7 — Web Application Frontend & Integration
**Week 7 | Status: TODO**

### Objectives
- [ ] Build React/HTML frontend with sequence input form
- [ ] Render predicted structure as color-coded sequence string
- [ ] Display confidence scores per residue (heatmap)
- [ ] Connect frontend to backend API
- [ ] End-to-end integration testing
- [ ] Responsive design for mobile

### UI Components
- Sequence input textarea (with validation)
- Submit button + loading indicator
- Color-coded prediction output:
  - `H (Helix)` → Red
  - `E (Strand)` → Yellow
  - `C (Coil)` → Blue
- Downloadable results (TXT/JSON)
- Example sequences for quick demo

---

## Phase 8 — Final Report & Presentation
**Week 8 | Status: TODO**

### Objectives
- [ ] Write final project report (IEEE/ACM format)
- [ ] Prepare slides for 15-minute presentation
- [ ] Record demo video of web application
- [ ] Submit code with README and reproducibility instructions
- [ ] Final GitHub cleanup (documentation, tests, CI)

### Report Sections
1. Abstract
2. Introduction & Motivation
3. Related Work (PSIPRED, DeepCNF, SPOT-1D, NetSurfP-3)
4. Dataset & Feature Engineering
5. Model Architectures (Bi-RNN & Bi-LSTM)
6. Training Methodology
7. Results & Analysis
8. Ablation Study
9. Web Application
10. Conclusion & Future Work
11. References

### Final Deliverables
- [ ] Source code (GitHub repository)
- [ ] Trained model checkpoints
- [ ] Reproducibility instructions (README.md)
- [ ] Final report (PDF)
- [ ] Presentation slides
- [ ] Demo video

---

## Phase Dependency Graph

```
Phase 1 (Framing)
    │
    ▼
Phase 2 (Data Pipeline) ─────────────────────┐
    │                                         │
    ▼                                         ▼
Phase 3 (Bi-RNN Baseline)            Phase 4 (Bi-LSTM)
    │                                         │
    └─────────────────┬───────────────────────┘
                      ▼
              Phase 5 (Analysis)
                      │
                      ▼
              Phase 6 (Backend)
                      │
                      ▼
              Phase 7 (Frontend + Integration)
                      │
                      ▼
              Phase 8 (Report)
```
