"""
Unit tests for accuracy and metric utilities.

Run with: pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.metrics import (
    masked_accuracy,
    compute_q3_accuracy,
    compute_q8_accuracy,
    compute_confusion_matrix,
    compute_per_class_f1,
    EpochMetrics,
)


class TestMaskedAccuracy:
    """Tests for the core masked_accuracy function."""

    def test_perfect_predictions(self):
        """All correct predictions should yield accuracy = 1.0."""
        preds  = torch.tensor([[0, 1, 2, 3]])
        labels = torch.tensor([[0, 1, 2, 3]])
        mask   = torch.tensor([[True, True, True, True]])
        assert masked_accuracy(preds, labels, mask) == 1.0

    def test_all_wrong_predictions(self):
        """All incorrect predictions should yield accuracy = 0.0."""
        preds  = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[0, 0, 0, 0]])
        mask   = torch.tensor([[True, True, True, True]])
        assert masked_accuracy(preds, labels, mask) == 0.0

    def test_half_correct(self):
        """50% correct predictions should yield accuracy ≈ 0.5."""
        # Positions 0,2 are correct; positions 1,3 are wrong → 2/4 = 0.5
        preds  = torch.tensor([[0, 9, 2, 9]])
        labels = torch.tensor([[0, 1, 2, 3]])
        mask   = torch.tensor([[True, True, True, True]])
        acc = masked_accuracy(preds, labels, mask)
        assert abs(acc - 0.5) < 1e-6

    def test_padding_excluded(self):
        """Padded positions must not contribute to accuracy."""
        preds  = torch.tensor([[0, 0, 99, 99]])   # positions 2,3 are garbage
        labels = torch.tensor([[0, 0, -1, -1]])
        mask   = torch.tensor([[True, True, False, False]])
        acc = masked_accuracy(preds, labels, mask)
        assert acc == 1.0, "Padded positions must be ignored"

    def test_empty_mask_returns_zero(self):
        """All-False mask (no real residues) must return 0.0, not division error."""
        preds  = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[0, 1, 2]])
        mask   = torch.tensor([[False, False, False]])
        assert masked_accuracy(preds, labels, mask) == 0.0

    def test_batch_of_sequences(self):
        """Accuracy must aggregate correctly across all sequences in a batch."""
        # 2 sequences of length 4, 3 correct out of 6 real residues
        preds  = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
        labels = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]])
        mask   = torch.tensor([[True, True, True, False],   # 3 real, 2 correct
                                [True, True, False, False]]) # 2 real, 2 correct
        # correct: 2 + 2 = 4, total real: 3 + 2 = 5
        acc = masked_accuracy(preds, labels, mask)
        assert abs(acc - 4/5) < 1e-6


class TestQ3Q8Accuracy:
    """Tests for the Q3/Q8 accuracy wrappers."""

    def test_q3_accuracy_delegates_correctly(self):
        preds  = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[0, 1, 2]])
        mask   = torch.ones(1, 3, dtype=torch.bool)
        assert compute_q3_accuracy(preds, labels, mask) == 1.0

    def test_q8_accuracy_delegates_correctly(self):
        preds  = torch.tensor([[0, 7, 3]])
        labels = torch.tensor([[0, 7, 3]])
        mask   = torch.ones(1, 3, dtype=torch.bool)
        assert compute_q8_accuracy(preds, labels, mask) == 1.0


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_perfect_confusion_matrix_is_diagonal(self):
        """Perfect predictions must yield a diagonal confusion matrix."""
        preds  = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[0, 1, 2]])
        mask   = torch.ones(1, 3, dtype=torch.bool)
        cm = compute_confusion_matrix(preds, labels, mask, num_classes=3)
        assert cm.shape == (3, 3)
        assert np.all(cm == np.diag(np.diag(cm)))

    def test_confusion_matrix_sum_equals_num_real_residues(self):
        """Total count in CM must equal number of unmasked residues."""
        preds  = torch.tensor([[0, 1, 2, 0]])
        labels = torch.tensor([[0, 0, 2, -1]])
        mask   = torch.tensor([[True, True, True, False]])
        cm = compute_confusion_matrix(preds, labels, mask, num_classes=3)
        assert cm.sum() == 3  # only 3 real residues

    def test_confusion_matrix_off_diagonal(self):
        """Misclassifications must appear in the correct off-diagonal cell."""
        preds  = torch.tensor([[1]])   # predicted class 1
        labels = torch.tensor([[0]])   # true class 0
        mask   = torch.tensor([[True]])
        cm = compute_confusion_matrix(preds, labels, mask, num_classes=3)
        assert cm[0, 1] == 1   # true=0, pred=1
        assert cm[0, 0] == 0   # no correct class-0 predictions


class TestPerClassF1:
    """Tests for per-class F1 computation from confusion matrix."""

    def test_perfect_f1_is_one(self):
        """Perfect diagonal CM must yield F1=1.0 for all classes."""
        cm = np.diag([10, 10, 10]).astype(np.int64)
        f1 = compute_per_class_f1(cm, ["C", "H", "E"])
        for cls in ["C", "H", "E"]:
            assert abs(f1[cls]["f1"] - 1.0) < 1e-6

    def test_all_zeros_f1_is_zero(self):
        """All-zero CM (edge case) must not raise errors."""
        cm = np.zeros((3, 3), dtype=np.int64)
        f1 = compute_per_class_f1(cm, ["C", "H", "E"])
        for cls in ["C", "H", "E"]:
            assert f1[cls]["f1"] == 0.0


class TestEpochMetrics:
    """Tests for the EpochMetrics accumulator."""

    def test_compute_on_empty_returns_zeros(self):
        """Computing metrics before any updates must return zero values."""
        m = EpochMetrics()
        result = m.compute()
        assert result["loss"] == 0.0
        assert result["q8_accuracy"] == 0.0
        assert result["q3_accuracy"] == 0.0

    def test_update_and_compute(self):
        """Updating with perfect predictions must yield accuracy = 1.0."""
        m = EpochMetrics()
        preds = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[0, 1, 2]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        m.update(
            q8_preds=preds, q8_labels=labels,
            q3_preds=preds, q3_labels=labels,
            mask=mask, loss=0.5,
        )
        result = m.compute()
        assert result["q8_accuracy"] == 1.0
        assert result["q3_accuracy"] == 1.0
        assert result["loss"] == 0.5

    def test_reset_clears_state(self):
        """Calling reset() must clear all accumulated state."""
        m = EpochMetrics()
        preds = torch.tensor([[0, 1]])
        labels = torch.tensor([[0, 1]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        m.update(preds, labels, preds, labels, mask, loss=1.0)
        m.reset()
        result = m.compute()
        assert result["loss"] == 0.0
