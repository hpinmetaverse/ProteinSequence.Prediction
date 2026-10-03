"""
2-Layer Stacked Bidirectional LSTM model for Protein Secondary Structure Prediction.

Architecture:
    Input (B, L, 42)
        → Linear embedding projection (42 → embed_dim=128) + LayerNorm + Dropout
        → Bidirectional LSTM Layer 1 (hidden=256) → output (B, L, 512)
        → Dropout(0.3)
        → Bidirectional LSTM Layer 2 (hidden=256) → output (B, L, 512)
        → Dropout(0.3)
        → Q8 head:  Linear(512 → 8)   [primary classification target]
        → Q3 head:  Linear(512 → 3)   [auxiliary head for optional joint training]

Why LSTM over vanilla RNN (key improvements):
    1. **Gating mechanism**: The forget, input, and output gates allow the LSTM to
       selectively retain or discard information, learning which context is relevant
       for each residue's secondary structure assignment.

    2. **Cell state**: The cell state `c_t` acts as a long-range memory that can
       persist signals across hundreds of residues — critical for beta-sheet topology
       where strand partners can be far apart in sequence.

    3. **Vanishing gradient mitigation**: Gates maintain gradient flow through long
       sequences via additive cell state updates (∂L/∂c_t remains bounded).

    4. **Stacked layers**: Two BiLSTM layers enable hierarchical feature learning:
       Layer 1 captures local structural motifs (helical turns, strand edges).
       Layer 2 aggregates these into longer-range structural patterns.

    5. **Bidirectionality**: Each residue sees both N→C (forward) and C→N (backward)
       context, reflecting the physical reality that backbone conformation depends
       on neighbors from both sides.

References:
    - Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
    - Li & Yu (2016). Protein Secondary Structure Prediction Using Cascaded CNNs.
    - Wang et al. (2016). Protein Secondary Structure Prediction Using Deep CNNs
      and LSTMs. arXiv:1612.01064
"""

from __future__ import annotations

from dataclasses import dataclass

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
class BiLSTMConfig:
    """Hyperparameter configuration for the 2-layer stacked Bidirectional LSTM.

    Attributes:
        input_dim: Dimensionality of per-residue input features (default 42).
        embed_dim: Linear projection dimension before LSTMs (default 128).
        hidden_dim: Hidden units per LSTM direction per layer (default 256).
            Total output per position = hidden_dim * 2 = 512.
        num_layers: Number of stacked Bi-LSTM layers (default 2).
        num_q8_classes: Number of Q8 output classes (default 8).
        num_q3_classes: Number of Q3 output classes for auxiliary head (default 3).
        dropout: Dropout probability after embedding and between LSTM layers
            (default 0.3).
        bidirectional: Whether to use bidirectional LSTM (default True).
        use_auxiliary_q3_head: Whether to include a separate Q3 output head
            for optional joint training with auxiliary Q3 loss (default True).
    """

    input_dim: int = INPUT_DIM
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    num_q8_classes: int = NUM_Q8_CLASSES
    num_q3_classes: int = NUM_Q3_CLASSES
    dropout: float = 0.3
    bidirectional: bool = True
    use_auxiliary_q3_head: bool = True

    @property
    def lstm_output_dim(self) -> int:
        """Effective output dimension of each LSTM layer.

        Returns:
            hidden_dim * 2 if bidirectional else hidden_dim.
        """
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim


# ─── Model ────────────────────────────────────────────────────────────────────

class BiLSTM(BaseModel):
    """2-layer stacked Bidirectional LSTM for protein secondary structure prediction.

    This is the primary (enhanced) model in the PSSP project. It leverages
    LSTM gating mechanisms and stacked bidirectionality to capture long-range
    structural motifs that vanilla Bi-RNN cannot model reliably.

    The model outputs Q8 logits as the primary prediction target and optionally
    Q3 logits from an auxiliary head. At inference time, Q3 predictions can be
    derived either from the Q3 head directly or by mapping Q8 predictions via
    the standard DSSP Q8→Q3 reduction.

    Args:
        config: BiLSTMConfig dataclass holding all hyperparameters.

    Example:
        >>> cfg = BiLSTMConfig(embed_dim=128, hidden_dim=256, num_layers=2)
        >>> model = BiLSTM(cfg)
        >>> x = torch.randn(8, 300, 42)        # batch of 8, max_len=300
        >>> lengths = torch.randint(50, 300, (8,))
        >>> output = model(x, lengths)
        >>> print(output.q8_logits.shape)      # (8, 300, 8)
        >>> print(output.q3_logits.shape)      # (8, 300, 3)
    """

    def __init__(self, config: BiLSTMConfig | None = None) -> None:
        """Initialize the 2-layer stacked Bi-LSTM model.

        Args:
            config: BiLSTMConfig with model hyperparameters.
                Defaults to BiLSTMConfig() if not provided.
        """
        super().__init__(model_name="BiLSTM")
        self.config = config or BiLSTMConfig()
        cfg = self.config

        # ── Embedding projection ──────────────────────────────────────────────
        # Projects 42-dim feature vectors to embed_dim before LSTM processing.
        # LayerNorm stabilizes the input distribution across variable-length seqs.
        self.embedding = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.embed_dim, bias=True),
            nn.LayerNorm(cfg.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
        )

        # ── Stacked Bidirectional LSTM ────────────────────────────────────────
        # We use two SEPARATE nn.LSTM modules (each 1-layer bidirectional)
        # rather than a single 2-layer nn.LSTM, because this allows:
        #   1. Inserting dropout + normalization between layers explicitly
        #   2. Independent initialization per layer
        #   3. Easier ablation (swap one layer for another type)
        #
        # Layer 1: embed_dim → lstm_output_dim (e.g., 128 → 512)
        self.lstm_layer1 = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        # Between-layer regularization
        self.inter_layer_dropout = nn.Dropout(p=cfg.dropout)

        # Layer 2: lstm_output_dim → lstm_output_dim (e.g., 512 → 512)
        self.lstm_layer2 = nn.LSTM(
            input_size=cfg.lstm_output_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        # Post-LSTM dropout before output heads
        self.output_dropout = nn.Dropout(p=cfg.dropout)

        # ── Q8 output head ────────────────────────────────────────────────────
        # Primary prediction: maps concatenated BiLSTM hidden states → 8 classes
        self.q8_head = nn.Linear(cfg.lstm_output_dim, cfg.num_q8_classes)

        # ── Q3 auxiliary output head ──────────────────────────────────────────
        # Optional: maps same hidden states → 3 classes for joint training.
        # Can be used with auxiliary CE loss: total_loss = L_q8 + λ * L_q3
        self.q3_head: nn.Linear | None = (
            nn.Linear(cfg.lstm_output_dim, cfg.num_q3_classes)
            if cfg.use_auxiliary_q3_head
            else None
        )

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights for stable training.

        Strategy:
            - LSTM hidden-to-hidden (weight_hh): Orthogonal initialization
              helps preserve gradient norms through recurrent connections.
            - LSTM input-to-hidden (weight_ih): Kaiming uniform (He init).
            - LSTM biases: Zero init. The forget gate bias is set to +1.0
              to encourage the model to initially pass information (prevents
              early-training forgetting of all context).
            - Linear layers: Kaiming uniform for weights, zero for biases.
        """
        for lstm in [self.lstm_layer1, self.lstm_layer2]:
            for name, param in lstm.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:
                    nn.init.kaiming_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1.0 (Jozefowicz et al. 2015)
                    # LSTM bias layout: [forget, input, cell, output] × hidden_dim
                    n = param.size(0)
                    # Forget gate is the second quarter of the bias vector
                    param.data[n // 4: n // 2].fill_(1.0)

        for module in [self.q8_head, self.q3_head]:
            if module is not None:
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _run_lstm_layer(
        self,
        lstm: nn.LSTM,
        x: torch.Tensor,
        lengths: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Pack, run LSTM, and unpack for one LSTM layer.

        Args:
            lstm: The nn.LSTM module to apply.
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            lengths: Actual sequence lengths on CPU, shape (batch_size,).
            seq_len: Total (padded) sequence length to restore after unpacking.

        Returns:
            LSTM output tensor of shape (batch_size, seq_len, hidden_dim * 2).
        """
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = lstm(packed)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq_len,
        )
        return out

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Run the 2-layer stacked Bi-LSTM forward pass.

        Processes a padded batch of amino acid sequences through:
            1. Embedding projection (42 → 128)
            2. Bi-LSTM Layer 1 (128 → 512) with packing for efficiency
            3. Inter-layer dropout
            4. Bi-LSTM Layer 2 (512 → 512) with packing
            5. Output dropout
            6. Q8 classifier head: logits (B, L, 8)
            7. Q3 auxiliary head: logits (B, L, 3) if configured

        Args:
            x: Input feature tensor of shape (batch_size, seq_len, 42).
                Concatenated one-hot amino acid encoding + PSSM profile per residue.
            lengths: Integer tensor of actual (pre-padding) sequence lengths,
                shape (batch_size,). Sequences beyond their length are padding.

        Returns:
            ModelOutput containing:
                - q8_logits: (batch_size, seq_len, 8) — Q8 class logits
                - q3_logits: (batch_size, seq_len, 3) or None — Q3 class logits
                - q8_preds:  (batch_size, seq_len) — discrete Q8 class indices
                - q3_preds:  (batch_size, seq_len) — discrete Q3 class indices
        """
        batch_size, seq_len, _ = x.shape

        # ── 1. Embedding projection: (B, L, 42) → (B, L, embed_dim) ─────────
        embedded = self.embedding(x)

        # ── 2. Bi-LSTM Layer 1: (B, L, embed_dim) → (B, L, 512) ─────────────
        # Packing skips padded positions, improving training efficiency.
        lstm1_out = self._run_lstm_layer(
            self.lstm_layer1, embedded, lengths, seq_len
        )

        # ── 3. Inter-layer dropout ────────────────────────────────────────────
        lstm1_out = self.inter_layer_dropout(lstm1_out)

        # ── 4. Bi-LSTM Layer 2: (B, L, 512) → (B, L, 512) ───────────────────
        # Second layer captures higher-order, longer-range structural patterns.
        lstm2_out = self._run_lstm_layer(
            self.lstm_layer2, lstm1_out, lengths, seq_len
        )

        # ── 5. Output dropout ─────────────────────────────────────────────────
        lstm2_out = self.output_dropout(lstm2_out)

        # ── 6. Q8 classification head ─────────────────────────────────────────
        # (B, L, 512) → (B, L, 8)
        q8_logits = self.q8_head(lstm2_out)

        # ── 7. Q3 auxiliary head ──────────────────────────────────────────────
        # (B, L, 512) → (B, L, 3) or None
        q3_logits = self.q3_head(lstm2_out) if self.q3_head is not None else None

        # ── 8. Derive discrete predictions ───────────────────────────────────
        q8_preds = torch.argmax(q8_logits, dim=-1)  # (B, L)
        if q3_logits is not None:
            # Prefer explicit Q3 head predictions when available
            q3_preds = torch.argmax(q3_logits, dim=-1)
        else:
            # Fall back to mapping Q8 predictions through the standard reduction
            q3_preds = q8_preds_to_q3(q8_preds)

        return ModelOutput(
            q8_logits=q8_logits,
            q3_logits=q3_logits,
            q8_preds=q8_preds,
            q3_preds=q3_preds,
        )


# ─── Factory function ─────────────────────────────────────────────────────────

def build_bi_lstm(config: BiLSTMConfig | None = None) -> BiLSTM:
    """Construct a BiLSTM model from configuration.

    Args:
        config: BiLSTMConfig with model hyperparameters.
            Defaults to BiLSTMConfig() with standard settings.

    Returns:
        Initialized BiLSTM model with forget gate bias = 1.0.

    Example:
        >>> cfg = BiLSTMConfig(embed_dim=128, hidden_dim=256, num_layers=2)
        >>> model = build_bi_lstm(cfg)
        >>> print(model.summary())
    """
    return BiLSTM(config=config)
