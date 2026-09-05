# Engineering & Coding Rules
## Protein Secondary Structure Prediction (PSSP) Project

---

> These rules govern all code written for this project. They exist to ensure
> consistency, reproducibility, and maintainability across all team members.

---

## 1. Python & Style Rules

### 1.1 Language & Version
- **Python >= 3.10** is required. All type hints must use modern union syntax (`X | None` not `Optional[X]`).
- **PyTorch >= 2.0** for all deep learning components.

### 1.2 Code Style
- Follow **PEP 8** strictly. Use `black` (line length 88) for formatting and `ruff` for linting.
- **No unused imports**. All imports must be explicitly used.
- **Absolute imports** only. Never use relative imports (`from ..model import X` is forbidden).
- **Type annotations** are required for all function signatures (arguments and return types).

```python
# CORRECT
def compute_q3_accuracy(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    ...

# WRONG — missing type hints
def compute_q3_accuracy(preds, labels, mask):
    ...
```

### 1.3 Docstrings
- All public classes, functions, and methods must have **Google-style docstrings**.

```python
def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Run a forward pass through the Bi-LSTM model.

    Args:
        x: Input feature tensor of shape (batch_size, seq_len, input_dim).
        lengths: Integer tensor of actual sequence lengths, shape (batch_size,).

    Returns:
        Q8 logits tensor of shape (batch_size, seq_len, 8).
    """
```

### 1.4 Magic Numbers
- **No hardcoded magic numbers** in model or data code. All constants go into config files or named constants at module level.

```python
# CORRECT
NUM_Q8_CLASSES = 8
NUM_Q3_CLASSES = 3
INPUT_DIM = 42
PADDING_IDX = -1

# WRONG
output = nn.Linear(512, 8)   # where does 8 come from?
```

---

## 2. PyTorch-Specific Rules

### 2.1 Tensor Device Handling
- Never hardcode `"cuda"` or `"cpu"`. Always use a `device` variable passed from the training script or detected at runtime.

```python
# CORRECT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)

# WRONG
model = model.cuda()
```

### 2.2 Gradient Clipping
- Always apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`.

### 2.3 Random Seeds
- Set all random seeds at training entry point for reproducibility:

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### 2.4 Padding Masking
- **All loss computations must mask padding positions.** Use `ignore_index` in `nn.CrossEntropyLoss` or boolean mask multiplication.
- **All accuracy computations must mask padding positions.** Never compute accuracy over padded tokens.

### 2.5 Model State
- Use `model.train()` before training loop, `model.eval()` before validation/test.
- Wrap inference in `torch.no_grad()`.

---

## 3. Data Rules

### 3.1 No Data Leakage
- CullPDB training split and CB513 test set must never be mixed during training or preprocessing.
- No normalization statistics computed using test data.

### 3.2 Reproducible Splits
- If splitting CullPDB into train/val, use a fixed `random_state` and document the split ratio.
- Default split: **80% train / 20% validation** from CullPDB filtered set.

### 3.3 Feature Extraction
- Feature columns from CullPDB `.npy`:
  - One-hot amino acids: columns **0–20** (21 columns)
  - PSSM profile: columns **35–55** (21 columns)
  - Q8 labels: columns **22–29** (8 classes, argmax)
  - Q3 labels: columns **31–33** (3 classes, argmax)
  - NoSeq mask: column **30** (1 = padding, should be ignored)

### 3.4 Sequence Length
- Sequences are pre-padded to length 700 in CullPDB. Trim trailing NoSeq tokens before returning from Dataset.__getitem__().

---

## 4. Model Rules

### 4.1 Base Class
- All models must inherit from `BaseModel` (see `src/models/base_model.py`).
- `BaseModel` enforces: `forward()`, `predict()`, `count_parameters()`.

### 4.2 Configuration-Driven
- Model hyperparameters must be passed via a dataclass or dict — never hardcoded into the class definition.

```python
# CORRECT
@dataclass
class BiLSTMConfig:
    input_dim: int = 42
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    num_q8_classes: int = 8
    num_q3_classes: int = 3
    dropout: float = 0.3

# WRONG
class BiLSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(42, 256, 2, bidirectional=True)  # hardcoded
```

### 4.3 Named Modules
- All nn.Module sub-components must be named clearly in `__init__` to aid debugging.

---

## 5. Training Rules

### 5.1 Checkpointing
- Save checkpoints to `checkpoints/` directory.
- Checkpoint filename format: `{model_name}_best.pt`
- Save: model state dict, optimizer state dict, epoch number, best validation Q3.
- Load checkpoint with `strict=True` always.

### 5.2 Logging
- Print training progress per epoch in this format:
  ```
  Epoch 01/50 | Train Loss: 1.2345 | Train Q3: 65.23% | Val Q3: 63.12% | Val Q8: 58.45%
  ```
- Save full training history to `logs/{model_name}_history.json`.

### 5.3 Early Stopping
- Stop training if validation Q3 does not improve for `patience=10` consecutive epochs.

---

## 6. Testing Rules

### 6.1 Test Location
- All tests go in the `tests/` directory.
- Test files must be named `test_*.py`.
- Use `pytest` as the test runner.

### 6.2 What to Test
- **Model forward pass**: Check output shapes match expectations for various input shapes.
- **Dataset loading**: Check feature tensor shapes, dtype, and value ranges.
- **Collator**: Check output is properly padded and masked for heterogeneous batch.
- **Metrics**: Check accuracy returns 1.0 for perfect predictions and 0.0 for all-wrong.

### 6.3 Test Data
- Use **synthetic small tensors** for unit tests — do not depend on real dataset files being present.

---

## 7. Git & Version Control Rules

### 7.1 Commits
- Write descriptive commit messages in imperative mood: `Add Bi-LSTM model with auxiliary Q3 head`.
- Commit logical, atomic units. Do not commit broken or untested code to `main`.

### 7.2 Branches
- `main`: Stable, passing code only.
- `feature/<name>`: New features (e.g., `feature/bi-lstm-model`).
- `fix/<name>`: Bug fixes.

### 7.3 Files to Never Commit
- Dataset files (`data/*.npy`) — too large, add to `.gitignore`
- Checkpoint files (`checkpoints/*.pt`) — add to `.gitignore`
- `__pycache__/`, `.DS_Store`, `*.pyc`

---

## 8. Documentation Rules

### 8.1 README
- `README.md` must have: project description, setup instructions, training command, evaluation command.

### 8.2 Config Files
- Every model must have a corresponding YAML config in `configs/`.
- All hyperparameters must be documented with comments in the YAML.

### 8.3 Notebooks
- Exploratory notebooks go in `notebooks/` and should be numbered: `01_eda.ipynb`, `02_feature_analysis.ipynb`, etc.
- Never import custom modules via `sys.path.append` hacks — install the package in editable mode: `pip install -e .`
