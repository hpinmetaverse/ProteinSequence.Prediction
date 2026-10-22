"""
Unit tests for the DynamicPaddingCollator and its output properties.

Tests use synthetic tensors only — no real dataset files required.
Run with: pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.data.collator import DynamicPaddingCollator, build_dataloader


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_sample(seq_len: int, input_dim: int = 42) -> tuple[Tensor, Tensor, Tensor]:
    """Create a synthetic (features, q8_labels, q3_labels) tuple."""
    features = torch.randn(seq_len, input_dim)
    q8_labels = torch.randint(0, 8, (seq_len,))
    q3_labels = torch.randint(0, 3, (seq_len,))
    return features, q8_labels, q3_labels


# ─── DynamicPaddingCollator tests ─────────────────────────────────────────────

class TestDynamicPaddingCollator:
    """Tests for the custom dynamic padding collator."""

    @pytest.fixture
    def collator(self):
        return DynamicPaddingCollator(padding_value=0.0, label_pad_value=-1)

    def test_output_is_tuple_of_five(self, collator):
        """Collator must return exactly 5 tensors."""
        batch = [make_sample(30), make_sample(20), make_sample(50)]
        result = collator(batch)
        assert len(result) == 5

    def test_features_padded_to_max_len(self, collator):
        """Feature tensor must be padded to the longest sequence in the batch."""
        lengths = [30, 20, 50]
        batch = [make_sample(l) for l in lengths]
        features, _, _, _, _ = collator(batch)
        assert features.shape == (3, 50, 42)

    def test_labels_padded_to_max_len(self, collator):
        """Label tensors must be padded to the longest sequence in the batch."""
        batch = [make_sample(30), make_sample(20), make_sample(50)]
        _, q8, q3, _, _ = collator(batch)
        assert q8.shape == (3, 50)
        assert q3.shape == (3, 50)

    def test_lengths_tensor_correct(self, collator):
        """lengths tensor must reflect actual (pre-padding) sequence lengths."""
        batch = [make_sample(30), make_sample(20), make_sample(50)]
        _, _, _, lengths, _ = collator(batch)
        expected = torch.tensor([30, 20, 50])
        assert torch.equal(lengths, expected)

    def test_mask_shape(self, collator):
        """Boolean mask must have shape (B, max_len)."""
        batch = [make_sample(30), make_sample(20), make_sample(50)]
        _, _, _, lengths, mask = collator(batch)
        assert mask.shape == (3, 50)

    def test_mask_dtype_is_bool(self, collator):
        """Mask must be a boolean tensor."""
        batch = [make_sample(30), make_sample(20)]
        _, _, _, _, mask = collator(batch)
        assert mask.dtype == torch.bool

    def test_mask_true_for_real_residues(self, collator):
        """Mask must be True at positions < actual sequence length."""
        batch = [make_sample(20), make_sample(50)]
        _, _, _, lengths, mask = collator(batch)
        # First sequence: positions 0–19 are real, 20–49 are padding
        assert mask[0, :20].all()
        assert not mask[0, 20:].any()
        # Second sequence: all 50 positions are real
        assert mask[1, :50].all()

    def test_padding_label_value(self, collator):
        """Padded label positions must equal label_pad_value (-1)."""
        batch = [make_sample(20), make_sample(50)]
        _, q8, _, _, _ = collator(batch)
        # First sequence: positions 20–49 are padded
        assert (q8[0, 20:] == -1).all()

    def test_padding_feature_value(self, collator):
        """Padded feature positions must equal padding_value (0.0)."""
        batch = [make_sample(20), make_sample(50)]
        features, _, _, _, _ = collator(batch)
        # First sequence: positions 20–49 should be zeros
        assert (features[0, 20:, :] == 0.0).all()

    def test_real_positions_not_overwritten(self, collator):
        """Real (non-padded) positions must preserve original feature values."""
        # Use a deterministic feature tensor
        feat = torch.ones(20, 42)
        q8 = torch.zeros(20, dtype=torch.long)
        q3 = torch.zeros(20, dtype=torch.long)
        batch = [(feat, q8, q3), make_sample(50)]
        features, _, _, _, _ = collator(batch)
        assert (features[0, :20, :] == 1.0).all()

    def test_batch_size_one(self, collator):
        """Collator must work with a single-element batch."""
        batch = [make_sample(35)]
        features, q8, q3, lengths, mask = collator(batch)
        assert features.shape == (1, 35, 42)
        assert lengths[0] == 35
        assert mask.all()

    def test_uniform_length_batch(self, collator):
        """Batch where all sequences have the same length needs no padding."""
        batch = [make_sample(40) for _ in range(8)]
        features, _, _, lengths, mask = collator(batch)
        assert features.shape == (8, 40, 42)
        assert mask.all()  # No padding needed


# ─── build_dataloader tests ────────────────────────────────────────────────────

class TestBuildDataloader:
    """Tests for the build_dataloader factory function."""

    def test_returns_dataloader(self):
        """build_dataloader must return a torch DataLoader."""
        from torch.utils.data import TensorDataset, DataLoader

        # Minimal dataset of (features, q8, q3) tuples
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self): return 10
            def __getitem__(self, idx):
                return make_sample(30 + idx)

        ds = TinyDataset()
        loader = build_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        assert isinstance(loader, DataLoader)

    def test_dataloader_iterates(self):
        """DataLoader created by build_dataloader must yield valid batches."""
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self): return 8
            def __getitem__(self, idx):
                return make_sample(20 + idx * 2)

        ds = TinyDataset()
        loader = build_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False, shuffle=False)
        batch = next(iter(loader))
        features, q8, q3, lengths, mask = batch
        assert features.ndim == 3
        assert features.shape[-1] == 42
