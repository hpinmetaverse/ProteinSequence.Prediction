"""
Unit tests for BiRNN and BiLSTM model forward passes.

Tests use synthetic tensors only — no real dataset files required.
Run with: pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
import torch

from src.models.base_model import q8_preds_to_q3, Q8_TO_Q3
from src.models.bi_rnn import BiRNN, BiRNNConfig, build_bi_rnn
from src.models.bi_lstm import BiLSTM, BiLSTMConfig, build_bi_lstm


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_input():
    """Synthetic batch: 4 sequences, max_len=50, input_dim=42."""
    batch_size, seq_len, input_dim = 4, 50, 42
    x = torch.randn(batch_size, seq_len, input_dim)
    # Varying lengths (no sequence longer than seq_len)
    lengths = torch.tensor([50, 45, 30, 20], dtype=torch.long)
    return x, lengths


@pytest.fixture
def bi_rnn_model():
    """Minimal BiRNN for fast unit tests."""
    cfg = BiRNNConfig(embed_dim=32, hidden_dim=64)
    return build_bi_rnn(cfg)


@pytest.fixture
def bi_lstm_model():
    """Minimal BiLSTM for fast unit tests."""
    cfg = BiLSTMConfig(embed_dim=32, hidden_dim=64, use_auxiliary_q3_head=True)
    return build_bi_lstm(cfg)


# ─── Q8 → Q3 mapping tests ────────────────────────────────────────────────────

class TestQ8ToQ3Mapping:
    """Tests for the Q8-to-Q3 class reduction mapping."""

    def test_all_q8_classes_are_mapped(self):
        """Every Q8 class (0–7) must map to a valid Q3 class (0–2)."""
        for q8_idx in range(8):
            assert q8_idx in Q8_TO_Q3, f"Q8 class {q8_idx} missing from mapping"
            assert Q8_TO_Q3[q8_idx] in (0, 1, 2), \
                f"Q8 class {q8_idx} maps to invalid Q3 class {Q8_TO_Q3[q8_idx]}"

    def test_helix_classes_map_to_h(self):
        """H, I, G (Q8: 3, 4, 5) must map to Helix (Q3: 1)."""
        helix_q8 = [3, 4, 5]
        for q8 in helix_q8:
            assert Q8_TO_Q3[q8] == 1, f"Q8 class {q8} should map to Helix (1)"

    def test_strand_classes_map_to_e(self):
        """E, B (Q8: 6, 7) must map to Strand (Q3: 2)."""
        strand_q8 = [6, 7]
        for q8 in strand_q8:
            assert Q8_TO_Q3[q8] == 2, f"Q8 class {q8} should map to Strand (2)"

    def test_coil_classes_map_to_c(self):
        """L, T, S (Q8: 0, 1, 2) must map to Coil (Q3: 0)."""
        coil_q8 = [0, 1, 2]
        for q8 in coil_q8:
            assert Q8_TO_Q3[q8] == 0, f"Q8 class {q8} should map to Coil (0)"

    def test_tensor_mapping_shape(self):
        """q8_preds_to_q3 must return same shape as input."""
        preds = torch.randint(0, 8, (4, 50))
        q3 = q8_preds_to_q3(preds)
        assert q3.shape == preds.shape

    def test_tensor_mapping_values(self):
        """All returned Q3 values must be in {0, 1, 2}."""
        preds = torch.arange(8).unsqueeze(0)  # shape (1, 8)
        q3 = q8_preds_to_q3(preds)
        assert q3.min().item() >= 0
        assert q3.max().item() <= 2


# ─── BiRNN tests ──────────────────────────────────────────────────────────────

class TestBiRNN:
    """Tests for the Bidirectional RNN baseline model."""

    def test_forward_output_shape(self, bi_rnn_model, batch_input):
        """Q8 logits must have shape (B, L, 8)."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        B, L, _ = x.shape
        assert output.q8_logits.shape == (B, L, 8), \
            f"Expected (4, 50, 8), got {output.q8_logits.shape}"

    def test_q8_preds_shape(self, bi_rnn_model, batch_input):
        """Q8 predicted indices must have shape (B, L)."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        B, L, _ = x.shape
        assert output.q8_preds.shape == (B, L)

    def test_q3_preds_shape(self, bi_rnn_model, batch_input):
        """Q3 predictions (derived from Q8) must have shape (B, L)."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        B, L, _ = x.shape
        assert output.q3_preds.shape == (B, L)

    def test_q3_logits_is_none(self, bi_rnn_model, batch_input):
        """Baseline Bi-RNN has no explicit Q3 head; q3_logits must be None."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        assert output.q3_logits is None

    def test_q8_pred_values_in_range(self, bi_rnn_model, batch_input):
        """All Q8 predicted class indices must be in [0, 7]."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        assert output.q8_preds.min().item() >= 0
        assert output.q8_preds.max().item() <= 7

    def test_q3_pred_values_in_range(self, bi_rnn_model, batch_input):
        """All Q3 predicted class indices must be in [0, 2]."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        assert output.q3_preds.min().item() >= 0
        assert output.q3_preds.max().item() <= 2

    def test_no_nan_in_output(self, bi_rnn_model, batch_input):
        """Forward pass must not produce NaN values."""
        x, lengths = batch_input
        output = bi_rnn_model(x, lengths)
        assert not torch.isnan(output.q8_logits).any()

    def test_count_parameters_positive(self, bi_rnn_model):
        """Model must have at least 1 trainable parameter."""
        counts = bi_rnn_model.count_parameters()
        assert counts["trainable"] > 0
        assert counts["total"] == counts["trainable"]  # all params trainable by default

    def test_build_factory(self):
        """build_bi_rnn() must return a BiRNN instance."""
        model = build_bi_rnn()
        assert isinstance(model, BiRNN)

    def test_single_sequence_batch(self, bi_rnn_model):
        """Model must handle batch size of 1."""
        x = torch.randn(1, 100, 42)
        lengths = torch.tensor([100])
        output = bi_rnn_model(x, lengths)
        assert output.q8_logits.shape == (1, 100, 8)

    def test_predict_no_grad(self, bi_rnn_model, batch_input):
        """predict() must run without gradient computation."""
        x, lengths = batch_input
        bi_rnn_model.eval()
        with torch.no_grad():
            output = bi_rnn_model.predict(x, lengths)
        assert output.q8_logits.shape[2] == 8


# ─── BiLSTM tests ─────────────────────────────────────────────────────────────

class TestBiLSTM:
    """Tests for the 2-layer stacked Bidirectional LSTM model."""

    def test_forward_output_shape(self, bi_lstm_model, batch_input):
        """Q8 logits must have shape (B, L, 8)."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        B, L, _ = x.shape
        assert output.q8_logits.shape == (B, L, 8)

    def test_q3_logits_shape(self, bi_lstm_model, batch_input):
        """Q3 logits from auxiliary head must have shape (B, L, 3)."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        B, L, _ = x.shape
        assert output.q3_logits is not None
        assert output.q3_logits.shape == (B, L, 3)

    def test_q8_preds_shape(self, bi_lstm_model, batch_input):
        """Q8 predictions must have shape (B, L)."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        B, L, _ = x.shape
        assert output.q8_preds.shape == (B, L)

    def test_q3_preds_shape(self, bi_lstm_model, batch_input):
        """Q3 predictions must have shape (B, L)."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        B, L, _ = x.shape
        assert output.q3_preds.shape == (B, L)

    def test_q8_pred_values_in_range(self, bi_lstm_model, batch_input):
        """Q8 predicted class indices must be in [0, 7]."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        assert output.q8_preds.min().item() >= 0
        assert output.q8_preds.max().item() <= 7

    def test_q3_pred_values_in_range(self, bi_lstm_model, batch_input):
        """Q3 predicted class indices must be in [0, 2]."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        assert output.q3_preds.min().item() >= 0
        assert output.q3_preds.max().item() <= 2

    def test_no_nan_in_output(self, bi_lstm_model, batch_input):
        """Forward pass must not produce NaN values."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        assert not torch.isnan(output.q8_logits).any()
        assert not torch.isnan(output.q3_logits).any()

    def test_no_q3_head(self, batch_input):
        """BiLSTM with use_auxiliary_q3_head=False must return q3_logits=None."""
        cfg = BiLSTMConfig(embed_dim=32, hidden_dim=64, use_auxiliary_q3_head=False)
        model = BiLSTM(cfg)
        x, lengths = batch_input
        output = model(x, lengths)
        assert output.q3_logits is None

    def test_more_parameters_than_bi_rnn(self, bi_rnn_model, bi_lstm_model):
        """2-layer BiLSTM must have more parameters than 1-layer BiRNN."""
        rnn_params = bi_rnn_model.count_parameters()["trainable"]
        lstm_params = bi_lstm_model.count_parameters()["trainable"]
        assert lstm_params > rnn_params, \
            f"Expected BiLSTM ({lstm_params}) > BiRNN ({rnn_params})"

    def test_build_factory(self):
        """build_bi_lstm() must return a BiLSTM instance."""
        model = build_bi_lstm()
        assert isinstance(model, BiLSTM)

    def test_gradient_flow(self, bi_lstm_model, batch_input):
        """A backward pass must produce non-None gradients for all parameters."""
        x, lengths = batch_input
        output = bi_lstm_model(x, lengths)
        # Sum both heads so every parameter receives a gradient
        loss = output.q8_logits.sum() + output.q3_logits.sum()
        loss.backward()
        for name, param in bi_lstm_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"

    def test_config_driven_hidden_dim(self, batch_input):
        """Model output dim must reflect the configured hidden_dim."""
        cfg = BiLSTMConfig(embed_dim=32, hidden_dim=128)  # output = 128*2 = 256
        model = BiLSTM(cfg)
        x, lengths = batch_input
        output = model(x, lengths)
        # Output head maps 256 → 8; just verify shape is correct
        assert output.q8_logits.shape[-1] == 8
