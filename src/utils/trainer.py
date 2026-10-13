"""
Training loop with best-validation checkpointing and early stopping.

The Trainer class encapsulates:
    1. Full training loop (multi-epoch)
    2. Masked CrossEntropy loss (ignoring padding tokens)
    3. Gradient clipping (max_norm=1.0)
    4. LR scheduler (ReduceLROnPlateau on val Q3)
    5. Best-checkpoint saving (by validation Q3 accuracy)
    6. Early stopping (patience=10 epochs)
    7. Training history logging

Usage:
    from src.utils.trainer import Trainer, TrainerConfig
    from src.models import build_bi_lstm

    model = build_bi_lstm()
    trainer = Trainer(model, train_loader, val_loader, config=TrainerConfig())
    history = trainer.fit()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel, ModelOutput
from src.utils.metrics import EpochMetrics


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    """Configuration for the training loop.

    Attributes:
        learning_rate: Initial Adam learning rate (default 1e-3).
        weight_decay: L2 regularization coefficient (default 1e-5).
        max_epochs: Maximum number of training epochs (default 50).
        grad_clip_max_norm: Maximum gradient norm for clipping (default 1.0).
        early_stopping_patience: Stop training if val Q3 does not improve
            for this many consecutive epochs (default 10).
        lr_scheduler_patience: ReduceLROnPlateau patience (default 3).
        lr_scheduler_factor: LR reduction factor (default 0.5).
        label_pad_value: Padding index to ignore in CrossEntropyLoss (default -1).
        checkpoint_dir: Directory to save model checkpoints (default 'checkpoints').
        log_dir: Directory to save training history JSON (default 'logs').
        device: Torch device string, auto-detected if None (default None).
        q8_loss_weight: Weight for Q8 CE loss component (default 1.0).
        q3_loss_weight: Weight for auxiliary Q3 CE loss (default 0.3).
            Set to 0.0 to disable auxiliary Q3 loss.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    grad_clip_max_norm: float = 1.0
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    label_pad_value: int = -1
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str | None = None
    q8_loss_weight: float = 1.0
    q3_loss_weight: float = 0.3


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """Manages the full training lifecycle for a PSSP sequence model.

    Handles training epochs, validation epochs, loss computation with
    proper padding masking, gradient clipping, LR scheduling, checkpointing,
    and early stopping.

    Args:
        model: A BaseModel subclass (BiRNN or BiLSTM).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: TrainerConfig with all training hyperparameters.
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The PyTorch model to train.
            train_loader: Training DataLoader (DynamicPaddingCollator expected).
            val_loader: Validation DataLoader (DynamicPaddingCollator expected).
            config: TrainerConfig. Defaults to TrainerConfig() if not provided.
        """
        self.config = config or TrainerConfig()
        cfg = self.config

        # ── Device setup ──────────────────────────────────────────────────────
        if cfg.device is not None:
            self.device = torch.device(cfg.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── Loss functions ────────────────────────────────────────────────────
        # ignore_index must match the label_pad_value used in DynamicPaddingCollator
        self.q8_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.label_pad_value,
            reduction="mean",
        )
        self.q3_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.label_pad_value,
            reduction="mean",
        )

        # ── Optimizer and scheduler ───────────────────────────────────────────
        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",        # maximize val Q3 accuracy
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.best_val_q3: float = 0.0
        self.epochs_without_improvement: int = 0
        self.start_epoch: int = 1
        self.history: list[dict[str, float]] = []

        # ── Create output directories ─────────────────────────────────────────
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    def _compute_loss(self, output: ModelOutput, q8_labels: Tensor, q3_labels: Tensor) -> Tensor:
        """Compute the total training loss.

        Primary loss: CrossEntropy on Q8 logits.
        Auxiliary loss: CrossEntropy on Q3 logits (if model has Q3 head).
        Total loss = q8_loss_weight * L_q8 + q3_loss_weight * L_q3

        Args:
            output: ModelOutput from the model's forward pass.
            q8_labels: Ground-truth Q8 labels, shape (B, L). Padded with ignore_index.
            q3_labels: Ground-truth Q3 labels, shape (B, L). Padded with ignore_index.

        Returns:
            Scalar loss tensor.
        """
        cfg = self.config

        # Reshape logits for CrossEntropyLoss: (B, C, L) expected
        # q8_logits shape: (B, L, 8) → transpose to (B, 8, L)
        q8_logits_t = output.q8_logits.permute(0, 2, 1)
        q8_loss = self.q8_criterion(q8_logits_t, q8_labels)

        total_loss = cfg.q8_loss_weight * q8_loss

        # Add auxiliary Q3 loss if the model provides a Q3 head
        if output.q3_logits is not None and cfg.q3_loss_weight > 0.0:
            q3_logits_t = output.q3_logits.permute(0, 2, 1)  # (B, 3, L)
            q3_loss = self.q3_criterion(q3_logits_t, q3_labels)
            total_loss = total_loss + cfg.q3_loss_weight * q3_loss

        return total_loss

    def _train_epoch(self) -> dict[str, float]:
        """Run one full training epoch over the training DataLoader.

        Returns:
            Dict with keys 'loss', 'q8_accuracy', 'q3_accuracy'.
        """
        self.model.train()
        epoch_metrics = EpochMetrics()

        for features, q8_labels, q3_labels, lengths, mask in self.train_loader:
            features = features.to(self.device)
            q8_labels = q8_labels.to(self.device)
            q3_labels = q3_labels.to(self.device)
            mask = mask.to(self.device)
            # lengths stays on CPU (required by pack_padded_sequence)

            # Forward pass
            self.optimizer.zero_grad()
            output: ModelOutput = self.model(features, lengths)

            # Compute masked loss
            loss = self._compute_loss(output, q8_labels, q3_labels)

            # Backward pass with gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_max_norm,
            )
            self.optimizer.step()

            epoch_metrics.update(
                q8_preds=output.q8_preds,
                q8_labels=q8_labels,
                q3_preds=output.q3_preds,
                q3_labels=q3_labels,
                mask=mask,
                loss=loss.item(),
            )

        return epoch_metrics.compute()

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        """Run one full validation epoch over the validation DataLoader.

        Returns:
            Dict with keys 'loss', 'q8_accuracy', 'q3_accuracy'.
        """
        self.model.eval()
        epoch_metrics = EpochMetrics()

        for features, q8_labels, q3_labels, lengths, mask in self.val_loader:
            features = features.to(self.device)
            q8_labels = q8_labels.to(self.device)
            q3_labels = q3_labels.to(self.device)
            mask = mask.to(self.device)

            output: ModelOutput = self.model(features, lengths)
            loss = self._compute_loss(output, q8_labels, q3_labels)

            epoch_metrics.update(
                q8_preds=output.q8_preds,
                q8_labels=q8_labels,
                q3_preds=output.q3_preds,
                q3_labels=q3_labels,
                mask=mask,
                loss=loss.item(),
            )

        return epoch_metrics.compute()

    def _save_checkpoint(self, epoch: int, val_q3: float) -> None:
        """Save the best model checkpoint to disk.

        Saves model state dict, optimizer state dict, epoch, and best val Q3
        so training can be resumed or the best model can be loaded for eval.

        Args:
            epoch: Current epoch number (1-indexed).
            val_q3: Validation Q3 accuracy at this epoch.
        """
        checkpoint_path = (
            Path(self.config.checkpoint_dir)
            / f"{self.model.model_name}_best.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_name": self.model.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_q3": val_q3,
                "history": self.history,
                "config": self.config,
            },
            checkpoint_path,
        )

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore training state from a saved checkpoint.

        Args:
            checkpoint_path: Path to a checkpoint produced by this trainer.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            ValueError: If the checkpoint belongs to a different model.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name != self.model.model_name:
            raise ValueError(
                f"Checkpoint is for {checkpoint_model_name!r}, but the current "
                f"model is {self.model.model_name!r}."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_q3 = float(checkpoint.get("best_val_q3", 0.0))
        self.start_epoch = int(checkpoint["epoch"]) + 1

        saved_history = checkpoint.get("history")
        if saved_history is not None:
            self.history = saved_history
        else:
            history_path = Path(self.config.log_dir) / (
                f"{self.model.model_name}_history.json"
            )
            if history_path.exists():
                with history_path.open() as history_file:
                    self.history = json.load(history_file)

        print(
            f"Resumed {self.model.model_name} from epoch {checkpoint['epoch']} "
            f"(best val Q3: {self.best_val_q3 * 100:.2f}%)."
        )

    def _save_history(self) -> None:
        """Write training history to a JSON file in the log directory."""
        log_path = (
            Path(self.config.log_dir)
            / f"{self.model.model_name}_history.json"
        )
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def fit(self) -> list[dict[str, float]]:
        """Run the full training loop for up to max_epochs epochs.

        Training stops early if validation Q3 accuracy does not improve
        for early_stopping_patience consecutive epochs.

        Returns:
            List of per-epoch metric dicts, each containing:
            'epoch', 'train_loss', 'train_q8', 'train_q3',
            'val_loss', 'val_q8', 'val_q3', 'lr', 'epoch_time_s'.
        """
        cfg = self.config
        print(f"\n{'='*70}")
        print(f"Training {self.model.model_name} on {self.device}")
        print(f"Parameters: {self.model.count_parameters()['trainable']:,} trainable")
        print(f"{'='*70}\n")

        for epoch in range(self.start_epoch, cfg.max_epochs + 1):
            t_start = time.time()

            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()

            elapsed = time.time() - t_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # LR scheduler step (based on val Q3)
            self.scheduler.step(val_metrics["q3_accuracy"])

            # Log epoch results
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "train_q8": round(train_metrics["q8_accuracy"] * 100, 2),
                "train_q3": round(train_metrics["q3_accuracy"] * 100, 2),
                "val_loss": round(val_metrics["loss"], 4),
                "val_q8": round(val_metrics["q8_accuracy"] * 100, 2),
                "val_q3": round(val_metrics["q3_accuracy"] * 100, 2),
                "lr": current_lr,
                "epoch_time_s": round(elapsed, 1),
            }
            self.history.append(epoch_record)

            print(
                f"Epoch {epoch:02d}/{cfg.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Q3: {train_metrics['q3_accuracy']*100:.2f}% | "
                f"Val Q3: {val_metrics['q3_accuracy']*100:.2f}% | "
                f"Val Q8: {val_metrics['q8_accuracy']*100:.2f}% | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing: save if val Q3 improved
            val_q3 = val_metrics["q3_accuracy"]
            if val_q3 > self.best_val_q3:
                self.best_val_q3 = val_q3
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_q3)
                print(f"  ✓ New best val Q3: {val_q3*100:.2f}% — checkpoint saved.")
            else:
                self.epochs_without_improvement += 1
                print(
                    f"  · No improvement ({self.epochs_without_improvement}/"
                    f"{cfg.early_stopping_patience})"
                )

            # Early stopping
            if self.epochs_without_improvement >= cfg.early_stopping_patience:
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Best val Q3: {self.best_val_q3*100:.2f}%"
                )
                break

        self._save_history()
        print(f"\nTraining complete. Best val Q3: {self.best_val_q3*100:.2f}%")
        return self.history

    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint back into the model.

        Raises:
            FileNotFoundError: If no checkpoint exists for this model.
        """
        checkpoint_path = (
            Path(self.config.checkpoint_dir)
            / f"{self.model.model_name}_best.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}. "
                "Run trainer.fit() first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(
            f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
            f"(val Q3 = {checkpoint['best_val_q3']*100:.2f}%)"
        )
