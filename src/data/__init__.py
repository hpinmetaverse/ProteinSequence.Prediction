"""
Data package for Protein Secondary Structure Prediction.

Provides:
    - ProteinDataset: PyTorch Dataset for CullPDB and CB513 data files
    - split_dataset:  Helper to split CullPDB into train/val subsets
    - DynamicPaddingCollator: Custom collate_fn for dynamic batch padding
    - build_dataloader: Factory for DataLoader with DynamicPaddingCollator
"""

from src.data.dataset import (
    ProteinDataset,
    CullPDBCSVDataset,
    load_cullpdb_csv_splits,
    split_dataset,
    INPUT_DIM,
    INPUT_DIM_CSV,
    INPUT_DIM_NPY,
    NUM_Q8_CLASSES,
    NUM_Q3_CLASSES,
    Q8_CLASS_NAMES,
    Q3_CLASS_NAMES,
)
from src.data.collator import DynamicPaddingCollator, build_dataloader

__all__ = [
    # Dataset classes
    "ProteinDataset",
    "CullPDBCSVDataset",
    "load_cullpdb_csv_splits",
    "split_dataset",
    # Collator
    "DynamicPaddingCollator",
    "build_dataloader",
    # Constants
    "INPUT_DIM",
    "INPUT_DIM_CSV",
    "INPUT_DIM_NPY",
    "NUM_Q8_CLASSES",
    "NUM_Q3_CLASSES",
    "Q8_CLASS_NAMES",
    "Q3_CLASS_NAMES",
]
