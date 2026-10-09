"""
Evaluation metrics for Protein Secondary Structure Prediction.

Provides masked Q3/Q8 accuracy, per-class F1 scores, and confusion matrices.
All metrics correctly ignore padded positions using a boolean mask.

Key principle:
    Padded positions (label_pad_value) MUST be excluded from all metric
    calculations. Including them would artificially inflate accuracy since
    the model trivially predicts padding correctly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def masked_accuracy(
    preds: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> float:
    """Compute per-residue accuracy, ignoring padded positions.

    Args:
        preds: Predicted class indices, shape (batch_size, seq_len).
        labels: Ground-truth class indices, shape (batch_size, seq_len).
            Padded positions should have value -1 (or any value, they are
            masked out by the mask argument).
        mask: Boolean tensor, shape (batch_size, seq_len).
            True at real residue positions, False at padded positions.

    Returns:
        Scalar float: fraction of correctly predicted residues (0.0–1.0).

    Example:
        >>> preds  = torch.tensor([[3, 0, 6, 0]])  # 0 = pad prediction (masked)
        >>> labels = torch.tensor([[3, 0, 6, -1]])
        >>> mask   = torch.tensor([[True, True, True, False]])
        >>> masked_accuracy(preds, labels, mask)
        1.0
    """
    correct = (preds == labels) & mask
    total_real = mask.sum().item()
    if total_real == 0:
        return 0.0
    return float(correct.sum().item()) / float(total_real)


def compute_q3_accuracy(
    q3_preds: Tensor,
    q3_labels: Tensor,
    mask: Tensor,
) -> float:
    """Compute Q3 (3-state) per-residue accuracy, ignoring padded positions.

    Args:
        q3_preds: Predicted Q3 class indices, shape (B, L). Values in {0,1,2}.
        q3_labels: Ground-truth Q3 class indices, shape (B, L). Values in {0,1,2}.
        mask: Boolean mask, shape (B, L). True for real residues.

    Returns:
        Scalar float: Q3 accuracy over all real residues in the batch.
    """
    return masked_accuracy(q3_preds, q3_labels, mask)


def compute_q8_accuracy(
    q8_preds: Tensor,
    q8_labels: Tensor,
    mask: Tensor,
) -> float:
    """Compute Q8 (8-state) per-residue accuracy, ignoring padded positions.

    Args:
        q8_preds: Predicted Q8 class indices, shape (B, L). Values in {0..7}.
        q8_labels: Ground-truth Q8 class indices, shape (B, L). Values in {0..7}.
        mask: Boolean mask, shape (B, L). True for real residues.

    Returns:
        Scalar float: Q8 accuracy over all real residues in the batch.
    """
    return masked_accuracy(q8_preds, q8_labels, mask)


def compute_confusion_matrix(
    preds: Tensor,
    labels: Tensor,
    mask: Tensor,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix over masked residues only.

    Args:
        preds: Predicted class indices, shape (B, L).
        labels: Ground-truth class indices, shape (B, L).
        mask: Boolean mask, shape (B, L). True for real residues.
        num_classes: Total number of classes (8 for Q8, 3 for Q3).

    Returns:
        Confusion matrix as numpy array, shape (num_classes, num_classes).
        Entry [i, j] = number of residues with true label i predicted as j.
    """
    preds_flat = preds[mask].cpu().numpy()
    labels_flat = labels[mask].cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels_flat, preds_flat):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def compute_per_class_f1(
    cm: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1 from a confusion matrix.

    Args:
        cm: Confusion matrix, shape (num_classes, num_classes).
        class_names: List of class name strings, length num_classes.

    Returns:
        Dict mapping class name to {'precision': float, 'recall': float, 'f1': float}.

    Example:
        >>> cm = np.array([[50, 5, 0], [3, 40, 2], [0, 1, 30]])
        >>> f1 = compute_per_class_f1(cm, ['C', 'H', 'E'])
    """
    results: dict[str, dict[str, float]] = {}
    num_classes = cm.shape[0]

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # predicted as c but not truly c
        fn = cm[c, :].sum() - tp   # truly c but predicted as something else

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[class_names[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # Macro-averaged F1
    macro_f1 = np.mean([v["f1"] for v in results.values()])
    results["macro_avg"] = {
        "precision": float(np.mean([v["precision"] for v in results.values() if "macro" not in results])),
        "recall": float(np.mean([v["recall"] for v in results.values() if "macro" not in results])),
        "f1": float(macro_f1),
    }

    return results


class EpochMetrics:
    """Accumulates per-batch predictions and labels across an epoch.

    Use this class to compute epoch-level metrics correctly — accumulate
    predictions across all batches, then compute metrics once at the end.
    This avoids the incorrect pattern of averaging per-batch accuracies
    (which is biased when batches have different effective sizes).

    Example:
        >>> metrics = EpochMetrics()
        >>> for features, q8_labels, q3_labels, lengths, mask in loader:
        ...     output = model(features, lengths)
        ...     loss = criterion(output.q8_logits, q8_labels)
        ...     metrics.update(output.q8_preds, q8_labels, output.q3_preds, q3_labels, mask, loss)
        >>> print(metrics.compute())
    """

    def __init__(self) -> None:
        """Initialize empty accumulators."""
        self._total_loss: float = 0.0
        self._num_batches: int = 0
        self._all_q8_preds: list[Tensor] = []
        self._all_q8_labels: list[Tensor] = []
        self._all_q3_preds: list[Tensor] = []
        self._all_q3_labels: list[Tensor] = []

    def update(
        self,
        q8_preds: Tensor,
        q8_labels: Tensor,
        q3_preds: Tensor,
        q3_labels: Tensor,
        mask: Tensor,
        loss: Tensor | float,
    ) -> None:
        """Accumulate one batch's predictions (masked, flattened to 1D).

        Args:
            q8_preds: Q8 predicted classes, shape (B, L).
            q8_labels: Q8 ground-truth classes, shape (B, L).
            q3_preds: Q3 predicted classes, shape (B, L).
            q3_labels: Q3 ground-truth classes, shape (B, L).
            mask: Boolean mask, shape (B, L). True = real residue.
            loss: Scalar loss value for this batch.
        """
        # Flatten and mask to extract only real residue predictions (1D)
        flat_mask = mask.detach().cpu().view(-1)
        self._all_q8_preds.append(q8_preds.detach().cpu().view(-1)[flat_mask])
        self._all_q8_labels.append(q8_labels.detach().cpu().view(-1)[flat_mask])
        self._all_q3_preds.append(q3_preds.detach().cpu().view(-1)[flat_mask])
        self._all_q3_labels.append(q3_labels.detach().cpu().view(-1)[flat_mask])
        self._total_loss += float(loss) if isinstance(loss, Tensor) else loss
        self._num_batches += 1

    def compute(self) -> dict[str, float]:
        """Compute epoch-level metrics from all accumulated batches.

        Returns:
            Dict with keys: 'loss', 'q8_accuracy', 'q3_accuracy'.
        """
        if self._num_batches == 0:
            return {"loss": 0.0, "q8_accuracy": 0.0, "q3_accuracy": 0.0}

        # All tensors are 1D (only real residues), safe to concatenate
        q8_preds = torch.cat(self._all_q8_preds, dim=0)
        q8_labels = torch.cat(self._all_q8_labels, dim=0)
        q3_preds = torch.cat(self._all_q3_preds, dim=0)
        q3_labels = torch.cat(self._all_q3_labels, dim=0)

        total = len(q8_preds)
        q8_acc = float((q8_preds == q8_labels).sum()) / total if total > 0 else 0.0
        q3_acc = float((q3_preds == q3_labels).sum()) / total if total > 0 else 0.0

        return {
            "loss": self._total_loss / self._num_batches,
            "q8_accuracy": q8_acc,
            "q3_accuracy": q3_acc,
        }

    def reset(self) -> None:
        """Reset all accumulators for the next epoch."""
        self.__init__()

