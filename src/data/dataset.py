"""
PyTorch Datasets for Protein Secondary Structure Prediction.

Two dataset classes are provided:

1. ProteinDataset  — loads the Princeton/CullPDB-format .npy files
   (CB513.npy shape (514, 700, 57) with one-hot + PSSM + Q8/Q3 labels).
   Used for: CB513 test set evaluation.

2. CullPDBCSVDataset — loads the CB6133 Kaggle CSV files (train/val/test.csv)
   which contain AA sequence strings and Q8 DSSP character label strings.
   No PSSM available → input is 21-dim one-hot only.
   Used for: training and validation (CullPDB split).

Dataset formats:
─────────────────────────────────────────────────────────────────
CullPDB CSVs (train.csv / val.csv / test.csv)
  Columns: sequence (str), ss (str)
  sequence: raw amino acid string  e.g. "ACDEFG..."
  ss:       Q8 DSSP character string e.g. "LHHEEST..."
  Q8 chars: L, T, S, H, I, G, E, B  (same as standard DSSP 8-class)
  Input features per residue: 21-dim one-hot amino acid (no PSSM)
  → input_dim = 21

CB513.npy  (Princeton ICML2014 format, reshaped to (514, 700, 57))
  Columns per residue (57 total):
    0–20  : one-hot amino acid     (21 values)
    21    : NoSeq padding indicator
    22–29 : Q8 labels (one-hot, 8 classes)
    30    : Q8 NoSeq
    31–33 : Q3 labels (one-hot, 3 classes)
    34    : Q3 NoSeq
    35–55 : PSSM evolutionary profile (21 values)
    56    : solvent accessibility (unused)
  Input features per residue: cols 0–20 + cols 35–55 = 42-dim
  → input_dim = 42
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# ─── CB6133 CSV constants ──────────────────────────────────────────────────────

# Standard amino acid alphabet (20 + X for unknown)
_AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWXY"
_AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ALPHABET)}
_NUM_AA: int = len(_AA_ALPHABET)  # 21

# Q8 DSSP character → integer index mapping
# Order: L=0, T=1, S=2, H=3, I=4, G=5, E=6, B=7
_Q8_CHAR_TO_IDX: dict[str, int] = {
    "L": 0,  # Loop / Irregular
    "T": 1,  # Turn
    "S": 2,  # Bend
    "H": 3,  # Alpha-Helix
    "I": 4,  # Pi-Helix
    "G": 5,  # 3-10 Helix
    "E": 6,  # Beta-Strand
    "B": 7,  # Beta-Bridge
}

# Q8 → Q3 reduction: C=0 (Coil), H=1 (Helix), E=2 (Strand)
_Q8_TO_Q3: dict[int, int] = {
    0: 0, 1: 0, 2: 0,  # L, T, S → Coil
    3: 1, 4: 1, 5: 1,  # H, I, G → Helix
    6: 2, 7: 2,        # E, B    → Strand
}
_Q8_TO_Q3_ARRAY = np.array([_Q8_TO_Q3[i] for i in range(8)], dtype=np.int64)

# ─── CB513 .npy column indices ─────────────────────────────────────────────────

_NPY_AA_START: int = 0
_NPY_AA_END: int = 21
_NPY_NOSEQ_COL: int = 21
_NPY_Q8_START: int = 22
_NPY_Q8_END: int = 30
_NPY_Q3_START: int = 31
_NPY_Q3_END: int = 34
_NPY_PSSM_START: int = 35
_NPY_PSSM_END: int = 56

# ─── Shared constants ─────────────────────────────────────────────────────────

NUM_Q8_CLASSES: int = 8
NUM_Q3_CLASSES: int = 3
INPUT_DIM_CSV: int = 21   # one-hot only (no PSSM)
INPUT_DIM_NPY: int = 42   # one-hot + PSSM
INPUT_DIM = INPUT_DIM_NPY  # default for backwards compatibility

Q8_CLASS_NAMES: list[str] = ["L", "T", "S", "H", "I", "G", "E", "B"]
Q3_CLASS_NAMES: list[str] = ["C", "H", "E"]


# ─── CSV Dataset (CB6133 Kaggle CSVs) ─────────────────────────────────────────

class CullPDBCSVDataset(Dataset):
    """Dataset for CB6133 Kaggle CSV files (train.csv / val.csv / test.csv).

    Each CSV row contains:
        - sequence: raw amino acid string (chars from ACDEFGHIKLMNPQRSTVWXY)
        - ss:       Q8 secondary structure label string (LTSHIGEB chars)

    Input features per residue: 21-dimensional one-hot amino acid encoding.
    (No PSSM available in this dataset format.)

    Q3 labels are automatically derived from Q8 via the standard DSSP mapping.

    Args:
        csv_path: Path to the CSV file (train.csv, val.csv, or test.csv).
        split: Human-readable split label for logging.

    Example:
        >>> ds = CullPDBCSVDataset("data/train.csv", split="train")
        >>> features, q8_labels, q3_labels = ds[0]
        >>> print(features.shape)   # (seq_len, 21)
        >>> print(q8_labels.shape)  # (seq_len,)
    """

    def __init__(self, csv_path: str | Path, split: str = "unknown") -> None:
        import pandas as pd

        self.csv_path = Path(csv_path)
        self.split = split

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = {"sequence", "ss"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV must have columns {required_cols}, found {df.columns.tolist()}"
            )

        # Drop rows where sequence length != label length (data integrity)
        mask = df["sequence"].str.len() == df["ss"].str.len()
        dropped = (~mask).sum()
        if dropped > 0:
            print(f"[{split}] Dropped {dropped} rows with mismatched seq/label lengths.")
        df = df[mask].reset_index(drop=True)

        self._sequences: list[str] = df["sequence"].tolist()
        self._labels: list[str] = df["ss"].tolist()

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve one-hot features and Q8/Q3 labels for one protein.

        Args:
            idx: Protein index.

        Returns:
            Tuple of (features, q8_labels, q3_labels):
                - features:  float32, shape (seq_len, 21)
                - q8_labels: int64,   shape (seq_len,)
                - q3_labels: int64,   shape (seq_len,)
        """
        seq = self._sequences[idx]
        ss = self._labels[idx]
        seq_len = len(seq)

        # ── One-hot amino acid features ───────────────────────────────────────
        features = np.zeros((seq_len, _NUM_AA), dtype=np.float32)
        for i, aa in enumerate(seq):
            aa_idx = _AA_TO_IDX.get(aa, _AA_TO_IDX["X"])  # unknown → X
            features[i, aa_idx] = 1.0

        # ── Q8 labels ─────────────────────────────────────────────────────────
        q8_labels = np.array(
            [_Q8_CHAR_TO_IDX.get(c, 0) for c in ss], dtype=np.int64
        )

        # ── Q3 labels (derived from Q8) ────────────────────────────────────────
        q3_labels = _Q8_TO_Q3_ARRAY[q8_labels]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(q8_labels, dtype=torch.long),
            torch.tensor(q3_labels, dtype=torch.long),
        )

    def get_length(self, idx: int) -> int:
        return len(self._sequences[idx])

    def input_dim(self) -> int:
        return _NUM_AA  # 21

    def __repr__(self) -> str:
        return (
            f"CullPDBCSVDataset(split='{self.split}', "
            f"num_proteins={len(self)}, path='{self.csv_path}')"
        )


# ─── NPY Dataset (CB513.npy Princeton format) ─────────────────────────────────

class ProteinDataset(Dataset):
    """PyTorch Dataset for CB513 in Princeton .npy format (shape 514×700×57).

    Loads CB513.npy reshaped to (N, 700, 57) and extracts:
        - Input features: one-hot (cols 0–20) + PSSM (cols 35–55) = 42 dims
        - Q8 labels: argmax of cols 22–29
        - Q3 labels: argmax of cols 31–33

    Trailing NoSeq (padding) positions are trimmed per protein.

    Args:
        data_path: Path to the .npy file. Raw shape (N, 39900) or (N, 700, 57).
        split: Human-readable split label for logging.

    Example:
        >>> ds = ProteinDataset("data/cb513.npy", split="test")
        >>> features, q8_labels, q3_labels = ds[0]
        >>> print(features.shape)   # (actual_len, 42)
    """

    def __init__(self, data_path: str | Path, split: str = "unknown") -> None:
        self.data_path = Path(data_path)
        self.split = split

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        raw: np.ndarray = np.load(str(self.data_path))

        # Handle flat format (N, 39900) → reshape to (N, 700, 57)
        if raw.ndim == 2 and raw.shape[1] == 39900:
            raw = raw.reshape(raw.shape[0], 700, 57)
        elif raw.ndim != 3 or raw.shape[1] != 700:
            raise ValueError(
                f"Unexpected shape {raw.shape}. Expected (N, 700, 57) or (N, 39900)."
            )

        self._raw = raw.astype(np.float32)
        self._num_proteins = raw.shape[0]
        self._lengths = self._compute_lengths(self._raw[:, :, _NPY_NOSEQ_COL])

    @staticmethod
    def _compute_lengths(noseq_flags: np.ndarray) -> np.ndarray:
        lengths = np.zeros(noseq_flags.shape[0], dtype=np.int64)
        for i in range(noseq_flags.shape[0]):
            positions = np.where(noseq_flags[i] == 1)[0]
            lengths[i] = positions[0] if len(positions) > 0 else noseq_flags.shape[1]
        return np.maximum(lengths, 1)

    def __len__(self) -> int:
        return self._num_proteins

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve 42-dim features and Q8/Q3 labels for one protein.

        Args:
            idx: Protein index.

        Returns:
            Tuple of (features, q8_labels, q3_labels):
                - features:  float32, shape (actual_len, 42)
                - q8_labels: int64,   shape (actual_len,)
                - q3_labels: int64,   shape (actual_len,)
        """
        protein = self._raw[idx, :self._lengths[idx]]  # (actual_len, 57)

        one_hot = protein[:, _NPY_AA_START:_NPY_AA_END]  # (L, 21)
        pssm = protein[:, _NPY_PSSM_START:_NPY_PSSM_END]  # (L, 21)
        features = np.concatenate([one_hot, pssm], axis=-1)  # (L, 42)

        q8_labels = protein[:, _NPY_Q8_START:_NPY_Q8_END].argmax(axis=-1).astype(np.int64)
        q3_labels = protein[:, _NPY_Q3_START:_NPY_Q3_END].argmax(axis=-1).astype(np.int64)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(q8_labels, dtype=torch.long),
            torch.tensor(q3_labels, dtype=torch.long),
        )

    def get_length(self, idx: int) -> int:
        return int(self._lengths[idx])

    def input_dim(self) -> int:
        return INPUT_DIM_NPY  # 42

    def __repr__(self) -> str:
        return (
            f"ProteinDataset(split='{self.split}', "
            f"num_proteins={self._num_proteins}, path='{self.data_path}')"
        )


# ─── Helper: load all CB6133 splits ───────────────────────────────────────────

def load_cullpdb_csv_splits(
    data_dir: str | Path = "data",
) -> tuple[CullPDBCSVDataset, CullPDBCSVDataset, CullPDBCSVDataset]:
    """Load train / val / test splits from the CB6133 Kaggle CSV files.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Example:
        >>> train, val, test = load_cullpdb_csv_splits("data")
        >>> print(len(train), len(val), len(test))   # 5600, 272, 256
    """
    d = Path(data_dir)
    return (
        CullPDBCSVDataset(d / "train.csv", split="train"),
        CullPDBCSVDataset(d / "val.csv",   split="val"),
        CullPDBCSVDataset(d / "test.csv",  split="test"),
    )


def split_dataset(
    data_path: str | Path,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple:
    """Split a .npy ProteinDataset into train/val subsets (kept for compatibility)."""
    full = ProteinDataset(data_path, split="full")
    n_train = int(len(full) * train_fraction)
    n_val = len(full) - n_train
    gen = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_val], generator=gen)
