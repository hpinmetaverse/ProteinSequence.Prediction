"""
Base model abstract class for all PSSP sequence-to-sequence models.

All models in this project inherit from BaseModel, which enforces
a consistent interface for forward passes, prediction, and parameter counting.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Container for model predictions.

    Attributes:
        q8_logits: Raw Q8 logits, shape (batch_size, seq_len, 8).
        q3_logits: Raw Q3 logits, shape (batch_size, seq_len, 3) or None.
        q8_preds: Argmax Q8 class indices, shape (batch_size, seq_len).
        q3_preds: Argmax Q3 class indices, shape (batch_size, seq_len).
    """

    q8_logits: torch.Tensor
    q3_logits: torch.Tensor | None
    q8_preds: torch.Tensor
    q3_preds: torch.Tensor


# Q8 label index → Q3 label index mapping
# Q8: L=0, T=1, S=2, H=3, I=4, G=5, E=6, B=7
# Q3: C=0 (Coil), H=1 (Helix), E=2 (Strand)
Q8_TO_Q3: dict[int, int] = {
    0: 0,  # L (Loop/Irregular)  → C (Coil)
    1: 0,  # T (Turn)            → C (Coil)
    2: 0,  # S (Bend)            → C (Coil)
    3: 1,  # H (Alpha-Helix)     → H (Helix)
    4: 1,  # I (Pi-Helix)        → H (Helix)
    5: 1,  # G (3-10 Helix)      → H (Helix)
    6: 2,  # E (Beta-Strand)     → E (Strand)
    7: 2,  # B (Beta-Bridge)     → E (Strand)
}

# Pre-built mapping tensor for fast GPU-accelerated Q8→Q3 conversion
_Q8_TO_Q3_TENSOR: torch.Tensor = torch.tensor(
    [Q8_TO_Q3[i] for i in range(8)], dtype=torch.long
)


def q8_preds_to_q3(q8_preds: torch.Tensor) -> torch.Tensor:
    """Convert Q8 prediction indices to Q3 prediction indices.

    Args:
        q8_preds: Integer tensor of Q8 class indices, any shape.

    Returns:
        Integer tensor of Q3 class indices, same shape as input.
    """
    mapping = _Q8_TO_Q3_TENSOR.to(q8_preds.device)
    return mapping[q8_preds]


class BaseModel(nn.Module, abc.ABC):
    """Abstract base class for all PSSP bidirectional sequence models.

    Subclasses must implement:
        - forward(x, lengths) → ModelOutput

    Provides:
        - predict(): masked argmax prediction from logits
        - count_parameters(): total and trainable parameter counts
        - summary(): human-readable model description
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the base model.

        Args:
            model_name: Human-readable name for logging and checkpointing.
        """
        super().__init__()
        self.model_name = model_name

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Run a forward pass through the model.

        Args:
            x: Input feature tensor of shape (batch_size, seq_len, input_dim).
                Features are 42-dimensional: 21 one-hot + 21 PSSM per residue.
            lengths: Integer tensor of actual (non-padded) sequence lengths,
                shape (batch_size,). Used for packing and masking.

        Returns:
            ModelOutput containing Q8 logits, Q3 logits (if applicable),
            and discrete Q8/Q3 predictions.
        """
        ...

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Run inference (no gradient tracking).

        Args:
            x: Input feature tensor, shape (batch_size, seq_len, input_dim).
            lengths: Actual sequence lengths, shape (batch_size,).

        Returns:
            ModelOutput with logits and discrete predictions.
        """
        self.eval()
        return self.forward(x, lengths)

    def count_parameters(self) -> dict[str, int]:
        """Count total and trainable model parameters.

        Returns:
            Dict with keys 'total' and 'trainable' parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def summary(self) -> str:
        """Return a human-readable summary of the model.

        Returns:
            Formatted string with model name, architecture, and parameter counts.
        """
        counts = self.count_parameters()
        lines = [
            f"Model: {self.model_name}",
            f"  Total parameters:     {counts['total']:,}",
            f"  Trainable parameters: {counts['trainable']:,}",
            f"  Architecture:\n{self}",
        ]
        return "\n".join(lines)
