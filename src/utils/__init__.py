"""
Utils package for Protein Secondary Structure Prediction.

Provides:
    - EpochMetrics, masked_accuracy, compute_q3_accuracy, compute_q8_accuracy,
      compute_confusion_matrix, compute_per_class_f1  (from metrics.py)
    - Trainer, TrainerConfig  (from trainer.py)
"""

from src.utils.metrics import (
    EpochMetrics,
    masked_accuracy,
    compute_q3_accuracy,
    compute_q8_accuracy,
    compute_confusion_matrix,
    compute_per_class_f1,
)
from src.utils.trainer import Trainer, TrainerConfig

__all__ = [
    # Metrics
    "EpochMetrics",
    "masked_accuracy",
    "compute_q3_accuracy",
    "compute_q8_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_f1",
    # Trainer
    "Trainer",
    "TrainerConfig",
]
