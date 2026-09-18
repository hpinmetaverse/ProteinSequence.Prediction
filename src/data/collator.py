"""
Dynamic padding collator for variable-length protein sequence batches.

Instead of padding every sequence to a global maximum length (700), this
collator pads each batch to the longest sequence *within that batch*.
This dramatically reduces wasted computation on short-sequence batches,
particularly in the early CullPDB split where many proteins are under 200 residues.

Usage:
    from src.data.collator import DynamicPaddingCollator
    from torch.utils.data import DataLoader

    collator = DynamicPaddingCollator(padding_value=0.0, label_pad_value=-1)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collator)

    for features, q8_labels, q3_labels, lengths, mask in loader:
        # features:   (B, max_len_in_batch, 42)
        # q8_labels:  (B, max_len_in_batch)   — padded with label_pad_value
        # q3_labels:  (B, max_len_in_batch)   — padded with label_pad_value
        # lengths:    (B,)                    — actual sequence lengths
        # mask:       (B, max_len_in_batch)   — True for real residues
        ...
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class DynamicPaddingCollator:
    """Callable collator that pads a batch to the longest sequence in the batch.

    Each call to this collator receives a list of (features, q8_labels, q3_labels)
    tuples (as returned by ProteinDataset.__getitem__) and returns batch tensors
    padded to the maximum sequence length found in that specific batch.

    This is the recommended approach for variable-length sequential data because:
    - Reduces total computation vs. global max-length padding
    - Enables more efficient GPU utilization per batch
    - Works seamlessly with pack_padded_sequence in model forward passes

    Args:
        padding_value: Value used to pad feature tensors (default 0.0).
        label_pad_value: Value used to pad label tensors. Should match the
            ignore_index used in nn.CrossEntropyLoss (default -1).
    """

    def __init__(
        self,
        padding_value: float = 0.0,
        label_pad_value: int = -1,
    ) -> None:
        """Initialize the collator.

        Args:
            padding_value: Scalar fill value for feature tensor padding.
            label_pad_value: Integer fill value for label tensor padding.
                Must match the ignore_index in the loss function.
        """
        self.padding_value = padding_value
        self.label_pad_value = label_pad_value

    def __call__(
        self,
        batch: list[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Collate a list of protein samples into a padded batch.

        Args:
            batch: List of (features, q8_labels, q3_labels) tuples where:
                - features:   float32 tensor, shape (seq_len_i, 42)
                - q8_labels:  int64 tensor,   shape (seq_len_i,)
                - q3_labels:  int64 tensor,   shape (seq_len_i,)
                Sequence lengths seq_len_i may differ across items.

        Returns:
            Tuple of five tensors:
                - features_padded: float32, shape (B, max_len, 42)
                    Padded with self.padding_value at positions beyond seq_len_i.
                - q8_labels_padded: int64, shape (B, max_len)
                    Padded with self.label_pad_value.
                - q3_labels_padded: int64, shape (B, max_len)
                    Padded with self.label_pad_value.
                - lengths: int64, shape (B,)
                    Actual (non-padded) sequence length for each protein.
                - mask: bool, shape (B, max_len)
                    True at real residue positions, False at padded positions.
        """
        features_list, q8_list, q3_list = zip(*batch)
        batch_size = len(features_list)

        # ── Compute max length in this batch ──────────────────────────────────
        lengths = torch.tensor(
            [f.shape[0] for f in features_list], dtype=torch.long
        )
        max_len = int(lengths.max().item())
        input_dim = features_list[0].shape[1]  # 42

        # ── Allocate padded output tensors ────────────────────────────────────
        features_padded = torch.full(
            (batch_size, max_len, input_dim),
            fill_value=self.padding_value,
            dtype=torch.float32,
        )
        q8_labels_padded = torch.full(
            (batch_size, max_len),
            fill_value=self.label_pad_value,
            dtype=torch.long,
        )
        q3_labels_padded = torch.full(
            (batch_size, max_len),
            fill_value=self.label_pad_value,
            dtype=torch.long,
        )

        # ── Copy actual sequence data into padded tensors ─────────────────────
        for i, (feat, q8, q3) in enumerate(zip(features_list, q8_list, q3_list)):
            seq_len = feat.shape[0]
            features_padded[i, :seq_len, :] = feat
            q8_labels_padded[i, :seq_len] = q8
            q3_labels_padded[i, :seq_len] = q3

        # ── Build boolean mask: True = real residue, False = padding ─────────
        # Shape: (B, max_len)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

        return features_padded, q8_labels_padded, q3_labels_padded, lengths, mask

    def __repr__(self) -> str:
        return (
            f"DynamicPaddingCollator("
            f"padding_value={self.padding_value}, "
            f"label_pad_value={self.label_pad_value})"
        )


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    label_pad_value: int = -1,
    pin_memory: bool | None = None,
) -> torch.utils.data.DataLoader:
    """Construct a DataLoader with the DynamicPaddingCollator.

    Args:
        dataset: ProteinDataset or Subset instance.
        batch_size: Number of proteins per batch (default 32).
        shuffle: Whether to shuffle the dataset each epoch (default True).
        num_workers: Number of worker processes for data loading (default 4).
        label_pad_value: Integer fill value for label padding; must match
            ignore_index in the loss function (default -1).
        pin_memory: Pin memory for faster CUDA transfer. If None, enabled only
            when CUDA is available; disabled for MPS and CPU (default None).

    Returns:
        DataLoader configured with dynamic padding collation.

    Example:
        >>> from src.data.dataset import ProteinDataset
        >>> ds = ProteinDataset("data/cb513.npy", split="test")
        >>> loader = build_dataloader(ds, batch_size=16, shuffle=False)
        >>> for feats, q8, q3, lengths, mask in loader:
        ...     print(feats.shape)   # (16, max_len_in_batch, 42)
        ...     break
    """
    collator = DynamicPaddingCollator(label_pad_value=label_pad_value)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        drop_last=False,
    )
