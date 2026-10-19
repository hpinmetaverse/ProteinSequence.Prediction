"""Evaluate a trained PSSP checkpoint on a CSV test split.

Usage:
    python scripts/evaluate.py --model bi_rnn --checkpoint checkpoints/BiRNN_best.pt
    python scripts/evaluate.py --model bi_lstm --checkpoint checkpoints/BiLSTM_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is on the path when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collator import build_dataloader
from src.data.dataset import CullPDBCSVDataset
from src.utils.metrics import EpochMetrics
from train import build_model


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a PSSP checkpoint on CSV test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["bi_rnn", "bi_lstm"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and print masked Q3 and Q8 test accuracy."""
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    model = build_model(args).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name != model.model_name:
        raise ValueError(
            f"Checkpoint is for {checkpoint_model_name!r}, but --model builds "
            f"{model.model_name!r}."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    test_dataset = CullPDBCSVDataset(
        Path(args.data_dir) / "test.csv",
        split="test",
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        label_pad_value=-1,
    )

    metrics = EpochMetrics()
    with torch.no_grad():
        for features, q8_labels, q3_labels, lengths, mask in test_loader:
            features = features.to(device)
            q8_labels = q8_labels.to(device)
            q3_labels = q3_labels.to(device)
            mask = mask.to(device)
            output = model(features, lengths)
            metrics.update(
                q8_preds=output.q8_preds,
                q8_labels=q8_labels,
                q3_preds=output.q3_preds,
                q3_labels=q3_labels,
                mask=mask,
                loss=0.0,
            )

    result = metrics.compute()
    print(f"Model: {model.model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Test proteins: {len(test_dataset)}")
    print(f"Test Q3 accuracy: {result['q3_accuracy'] * 100:.2f}%")
    print(f"Test Q8 accuracy: {result['q8_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()