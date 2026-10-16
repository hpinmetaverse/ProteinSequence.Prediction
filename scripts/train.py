"""
Training entry point for PSSP models.

Usage:
    # Train the Bi-LSTM (enhanced model) on CB6133 CSVs
    python scripts/train.py --model bi_lstm

    # Train the Bi-RNN (baseline model)
    python scripts/train.py --model bi_rnn

    # Override hyperparameters
    python scripts/train.py --model bi_lstm --max_epochs 30 --batch_size 16

Output:
    checkpoints/{model_name}_best.pt  — best validation checkpoint
    logs/{model_name}_history.json    — per-epoch training history
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ── Ensure project root is on the path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CullPDBCSVDataset, INPUT_DIM_CSV
from src.data.collator import build_dataloader
from src.models.bi_rnn import BiRNN, BiRNNConfig, build_bi_rnn
from src.models.bi_lstm import BiLSTM, BiLSTMConfig, build_bi_lstm
from src.utils.trainer import Trainer, TrainerConfig


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PSSP model (BiRNN or BiLSTM) on CullPDB CSV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, choices=["bi_rnn", "bi_lstm"],
                        default="bi_lstm", help="Model architecture.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing train.csv, val.csv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--q3_loss_weight", type=float, default=0.3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path from which to resume training.",
    )

    return parser.parse_args()


# ─── Model builder ────────────────────────────────────────────────────────────

def build_model(args: argparse.Namespace) -> BiRNN | BiLSTM:
    """Build model with input_dim=21 for CSV training (one-hot only)."""
    input_dim = INPUT_DIM_CSV  # 21 (one-hot amino acids, no PSSM in CSV)

    if args.model == "bi_rnn":
        cfg = BiRNNConfig(
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        return build_bi_rnn(cfg)
    else:
        cfg = BiLSTMConfig(
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_auxiliary_q3_head=True,
        )
        return build_bi_lstm(cfg)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Protein Secondary Structure Prediction — Training")
    print(f"  Model:  {args.model.upper()}")
    print(f"  Seed:   {args.seed}")
    print(f"{'='*60}\n")

    # ── Load CSV data splits ──────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_ds = CullPDBCSVDataset(data_dir / "train.csv", split="train")
    val_ds   = CullPDBCSVDataset(data_dir / "val.csv",   split="val")

    print(f"Train: {len(train_ds)} proteins")
    print(f"Val:   {len(val_ds)} proteins")

    train_loader = build_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, label_pad_value=-1,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, label_pad_value=-1,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args)
    print(model.summary())

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_cfg = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        grad_clip_max_norm=args.grad_clip,
        early_stopping_patience=args.early_stopping,
        q3_loss_weight=args.q3_loss_weight if args.model == "bi_lstm" else 0.0,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    trainer = Trainer(model, train_loader, val_loader, config=trainer_cfg)

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit()
    print(f"\nTraining complete. History saved to {args.log_dir}/")


if __name__ == "__main__":
    main()
