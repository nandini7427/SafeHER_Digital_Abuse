"""Decoupled training harness for SafeHER Model A neural detectors.

Strictly separated from inference code to prevent leakage of training-time
logic, optimizer states, or non-deterministic operations into production pipelines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.training.checkpointer import save_detector_checkpoint


@dataclass
class EpochMetrics:
    """Quantitative performance metrics for a single training/validation epoch."""

    epoch: int
    loss: float
    accuracy: float
    precision: float
    recall: float
    sample_count: int


class ModelATrainer:
    """Trainer for Model A (AI-generated vs. camera photograph classification)."""

    def __init__(
        self,
        model: BaseNeuralDetector,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """Initialize trainer with model, optimizer, and loss criterion.

        Args:
            model: BaseNeuralDetector instance to train.
            learning_rate: Initial learning rate for AdamW optimizer.
            weight_decay: L2 regularization coefficient.
            label_smoothing: Label smoothing factor in cross-entropy loss.
            device: Target computing device (defaults to CPU).
            seed: Random seed for deterministic reproducibility.
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.seed = seed

        torch.manual_seed(seed)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def train_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        """Execute one training epoch over the provided DataLoader."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        tp = 0
        fp = 0
        fn = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_y).sum().item()

            # Binary metrics for AI_GENERATED_IMAGE (class 1)
            tp += ((preds == 1) & (batch_y == 1)).sum().item()
            fp += ((preds == 1) & (batch_y == 0)).sum().item()
            fn += ((preds == 0) & (batch_y == 1)).sum().item()

        self.scheduler.step()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0

        return EpochMetrics(
            epoch=0,
            loss=round(avg_loss, 4),
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            sample_count=total_samples,
        )

    def evaluate(self, dataloader: DataLoader) -> EpochMetrics:
        """Evaluate model on validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch_y).sum().item()

                tp += ((preds == 1) & (batch_y == 1)).sum().item()
                fp += ((preds == 1) & (batch_y == 0)).sum().item()
                fn += ((preds == 0) & (batch_y == 1)).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0

        return EpochMetrics(
            epoch=0,
            loss=round(avg_loss, 4),
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            sample_count=total_samples,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> List[EpochMetrics]:
        """Run training loop across multiple epochs."""
        history: List[EpochMetrics] = []

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader)
            metrics.epoch = epoch
            history.append(metrics)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_metrics.epoch = epoch

        # Mark model as trained after completing verified training loop
        self.model.is_trained = True

        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"{self.model.get_metadata().architecture}_checkpoint.pt"
            save_detector_checkpoint(
                model=self.model,
                output_path=ckpt_path,
                extra_metadata={"is_trained": True, "epochs_completed": epochs},
            )

        return history
