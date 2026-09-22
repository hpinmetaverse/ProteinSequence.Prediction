"""
Models package for Protein Secondary Structure Prediction.

Available models:
    - BiRNN:  Bidirectional RNN baseline (1-layer)
    - BiLSTM: 2-layer stacked Bidirectional LSTM (enhanced model)

Usage:
    from src.models import BiRNN, BiLSTM, BiRNNConfig, BiLSTMConfig, build_bi_rnn, build_bi_lstm
"""

from src.models.base_model import BaseModel, ModelOutput, Q8_TO_Q3, q8_preds_to_q3
from src.models.bi_rnn import BiRNN, BiRNNConfig, build_bi_rnn
from src.models.bi_lstm import BiLSTM, BiLSTMConfig, build_bi_lstm

__all__ = [
    # Base classes / types
    "BaseModel",
    "ModelOutput",
    "Q8_TO_Q3",
    "q8_preds_to_q3",
    # Bi-RNN baseline
    "BiRNN",
    "BiRNNConfig",
    "build_bi_rnn",
    # Bi-LSTM enhanced
    "BiLSTM",
    "BiLSTMConfig",
    "build_bi_lstm",
]
