"""
Bidirectional RNN (Bi-RNN) baseline model for Protein Secondary Structure Prediction.

Architecture:
    Input (B, L, 42)
        → Linear embedding projection (42 → embed_dim=128)
        → LayerNorm + Dropout
        → Bidirectional RNN (1 layer, hidden=256)
        → Dropout
        → Linear output head (512 → 8) for Q8 classification

The bidirectional design allows each residue's prediction to be conditioned
on context from BOTH the N-terminus (forward pass) and C-terminus (backward pass),
which is critical because secondary structure depends on neighboring residues
in both directions.

This model serves as the **baseline** for comparison against the enhanced Bi-LSTM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.base_model import BaseModel, ModelOutput, q8_preds_to_q3

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_Q8_CLASSES: int = 8   # L, T, S, H, I, G, E, B
NUM_Q3_CLASSES: int = 3   # C (Coil), H (Helix), E (Strand)
INPUT_DIM: int = 42       # 21 one-hot amino acids + 21 PSSM columns


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BiRNNConfig:
    """Hyperparameter configuration for the Bidirectional RNN model.

    Attributes:
        input_dim: Dimensionality of per-residue input features (default 42).
        embed_dim: Projection dimension before the RNN (default 128).
        hidden_dim: Number of hidden units in each RNN direction (default 256).
            Total hidden size passed to output layer = hidden_dim * 2.
        num_layers: Number of stacked RNN layers (default 1 for baseline).
        num_q8_classes: Number of Q8 output classes (default 8).
        dropout: Dropout probability applied after embedding and RNN (default 0.3).
        bidirectional: Whether to use a bidirectional RNN (default True).
    """

    input_dim: int = INPUT_DIM
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    num_q8_classes: int = NUM_Q8_CLASSES
    dropout: float = 0.3
    bidirectional: bool = True

    @property
    def rnn_output_dim(self) -> int:
        """Effective output dimension of the RNN layer.

        Returns:
            hidden_dim * 2 if bidirectional else hidden_dim.
        """
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim


# ─── Model ────────────────────────────────────────────────────────────────────

class BiRNN(BaseModel):
    """Bidirectional RNN baseline for protein secondary structure prediction.

    This single-layer bidirectional RNN serves as the baseline model.
    Each residue's output is the concatenation of forward and backward hidden
    states, giving it access to both left (N→C) and right (C→N) context.

    Limitations vs Bi-LSTM:
        - Vanilla RNN suffers from vanishing gradients on long sequences (>50 residues).
        - No gating mechanism: cannot selectively forget irrelevant context.
        - Long-range structural motifs (e.g., beta-sheet hydrogen bond partners
          separated by 100+ residues) are hard to capture.

    Args:
        config: BiRNNConfig dataclass holding all hyperparameters.
    """

    def __init__(self, config: BiRNNConfig | None = None) -> None:
        """Initialize the Bi-RNN model.

        Args:
            config: BiRNNConfig with model hyperparameters.
                Defaults to BiRNNConfig() if not provided.
        """
        super().__init__(model_name="BiRNN")
        self.config = config or BiRNNConfig()
        cfg = self.config

        # ── Embedding projection ──────────────────────────────────────────────
        # Projects raw 42-dim feature vectors into a richer embed_dim space
        # before feeding into the RNN. LayerNorm stabilizes training.
        self.embedding = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.embed_dim, bias=True),
            nn.LayerNorm(cfg.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
        )

        # ── Bidirectional RNN ─────────────────────────────────────────────────
        # num_directions = 2 → forward + backward passes concatenated.
        # Output shape: (B, L, hidden_dim * num_directions)
        self.rnn = nn.RNN(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=0.0,  # no inter-layer dropout (single layer baseline)
        )

        # ── Post-RNN regularization ───────────────────────────────────────────
        self.rnn_dropout = nn.Dropout(p=cfg.dropout)

        # ── Output head: Q8 classifier ────────────────────────────────────────
        # Maps the concatenated bi-directional hidden state → 8 class logits
        self.q8_head = nn.Linear(cfg.rnn_output_dim, cfg.num_q8_classes)

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using recommended strategies per layer type.

        - Linear layers: Kaiming uniform (He initialization for ReLU activations)
        - RNN: Orthogonal initialization for hidden-to-hidden weights;
               Kaiming uniform for input-to-hidden weights
        - Biases: Zero initialization
        """
        for name, param in self.named_parameters():
            if "rnn.weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "rnn.weight_ih" in name:
                nn.init.kaiming_uniform_(param)
            elif "rnn.bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Run the Bi-RNN forward pass.

        Processes a batch of variable-length amino acid sequences. Sequences
        are dynamically packed to ignore padding during RNN computation,
        then padded back to batch shape.

        Args:
            x: Input feature tensor of shape (batch_size, seq_len, 42).
                Contains concatenated one-hot + PSSM features per residue.
            lengths: Integer tensor of actual sequence lengths (before padding),
                shape (batch_size,). Must be on CPU for pack_padded_sequence.

        Returns:
            ModelOutput with:
                - q8_logits: shape (batch_size, seq_len, 8)
                - q3_logits: None (Bi-RNN baseline has no explicit Q3 head)
                - q8_preds: shape (batch_size, seq_len)
                - q3_preds: shape (batch_size, seq_len) via Q8→Q3 mapping
        """
        batch_size, seq_len, _ = x.shape

        # ── 1. Embedding projection ───────────────────────────────────────────
        # (B, L, 42) → (B, L, embed_dim)
        embedded = self.embedding(x)

        # ── 2. Pack sequences to skip padded positions in RNN ────────────────
        # lengths must be on CPU (PyTorch requirement for pack_padded_sequence)
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,  # handles unsorted batches safely
        )

        # ── 3. Bidirectional RNN ──────────────────────────────────────────────
        # packed_output shape (after unpack): (B, L, hidden_dim * 2)
        packed_output, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len,  # restore to original padded length
        )

        # ── 4. Dropout ────────────────────────────────────────────────────────
        rnn_out = self.rnn_dropout(rnn_out)

        # ── 5. Q8 classification head ─────────────────────────────────────────
        # (B, L, hidden_dim * 2) → (B, L, 8)
        q8_logits = self.q8_head(rnn_out)

        # ── 6. Derive predictions ─────────────────────────────────────────────
        q8_preds = torch.argmax(q8_logits, dim=-1)         # (B, L)
        q3_preds = q8_preds_to_q3(q8_preds)               # (B, L)

        return ModelOutput(
            q8_logits=q8_logits,
            q3_logits=None,  # baseline model predicts Q3 via Q8 mapping only
            q8_preds=q8_preds,
            q3_preds=q3_preds,
        )


# ─── Factory function ─────────────────────────────────────────────────────────

def build_bi_rnn(config: BiRNNConfig | None = None) -> BiRNN:
    """Construct a BiRNN model from configuration.

    Args:
        config: BiRNNConfig with model hyperparameters.
            Defaults to BiRNNConfig() with standard settings.

    Returns:
        Initialized BiRNN model.

    Example:
        >>> cfg = BiRNNConfig(embed_dim=64, hidden_dim=128)
        >>> model = build_bi_rnn(cfg)
        >>> print(model.summary())
    """
    return BiRNN(config=config)
