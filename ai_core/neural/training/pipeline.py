"""Reproducible Neural Training Pipeline for SafeHER Model A.

Coordinates:
- Rigid configuration parsing and master random seed configuration.
- Manifest rights validation, test split isolation, and private evidence quarantine.
- Manifest-driven PyTorch Dataset and DataLoader creation with exclusion audit tracking.
- Model instantiation via architecture factory.
- AdamW optimizer, label-smoothed cross-entropy loss, and configurable LR schedulers.
- Epoch training and validation loops with gradient clipping.
- Metric tracking, early stopping, and structured JSONL logging.
- Checkpoint persistence, SafeTensors export, and state resumption.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai_core.datasets.loader import SafeHERDataset, create_safe_dataloader
from ai_core.datasets.manifest import (
    DatasetManifest,
    DatasetSplit,
    TargetUsageScope,
)
from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.factory import create_image_detector
from ai_core.neural.training.checkpoint_manager import CheckpointManager
from ai_core.neural.training.config import (
    EarlyStoppingMetric,
    LRSchedulerType,
    TrainingPipelineConfig,
    TrainingPipelineResult,
)
from ai_core.neural.training.early_stopping import EarlyStopping
from ai_core.neural.training.logger import StructuredTrainingLogger
from ai_core.neural.training.safety import (
    ManifestSafetyAudit,
    enforce_training_safety_and_rights,
)

logger = logging.getLogger(__name__)


def set_reproducible_seed(seed: int, num_threads: int = 2) -> None:
    """Configure deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, num_threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelATrainingPipeline:
    """Production-grade, reproducible training pipeline for SafeHER Model A architectures."""

    def __init__(
        self,
        config: TrainingPipelineConfig,
        manifest: DatasetManifest,
        root_dir: Optional[Union[str, Path]] = None,
        in_memory_images: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ModelATrainingPipeline.

        Args:
            config: TrainingPipelineConfig instance.
            manifest: DatasetManifest containing training and validation records.
            root_dir: Optional directory resolving relative image paths.
            in_memory_images: Optional in-memory image dict for synthetic fixtures/tests.

        Raises:
            ValueError: If manifest safety validation, rights validation, or leakage check fails.
        """
        self.config = config
        self.manifest = manifest
        self.root_dir = root_dir or config.root_dir

        # Normalize in-memory images mapping across both record_id and image_path_or_ref
        if in_memory_images is not None:
            normalized_in_mem: Dict[str, Any] = dict(in_memory_images)
            for r in self.manifest.records:
                if r.image_path_or_ref in in_memory_images and r.record_id not in normalized_in_mem:
                    normalized_in_mem[r.record_id] = in_memory_images[r.image_path_or_ref]
                elif r.record_id in in_memory_images and r.image_path_or_ref not in normalized_in_mem:
                    normalized_in_mem[r.image_path_or_ref] = in_memory_images[r.record_id]
            self.in_memory_images = normalized_in_mem
        else:
            self.in_memory_images = None

        # 1. Enforce master seed reproducibility
        set_reproducible_seed(self.config.seed, self.config.num_threads)

        # 2. Enforce manifest safety, commercial rights, and test split isolation
        if self.config.enforce_approved_manifest:
            self.safety_audit = enforce_training_safety_and_rights(
                manifest=self.manifest,
                target_workload=self.config.target_workload,
            )
        else:
            from ai_core.neural.training.safety import (
                compute_records_hash,
                compute_split_allocation_hash,
            )
            recs = self.manifest.records
            self.safety_audit = ManifestSafetyAudit(
                is_safe=True,
                training_manifest_hash=compute_records_hash(recs),
                split_manifest_hash=compute_split_allocation_hash(recs),
                dataset_versions=sorted(list({r.dataset_version for r in recs if r.dataset_version})),
                train_record_count=len([r for r in recs if r.split == DatasetSplit.TRAIN]),
                val_record_count=len([r for r in recs if r.split == DatasetSplit.VAL]),
                test_record_count=len([r for r in recs if r.split == DatasetSplit.INTERNAL_TEST]),
                commercial_approved=False,
                quarantined_count=0,
                notes=["Manifest safety validation bypassed (enforce_approved_manifest=False)."],
            )

        # 3. Determine compute device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Mixed precision guard: only permitted on CUDA
        self.use_amp = self.config.use_mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None

        # 4. Instantiate Datasets and DataLoaders
        self.train_dataset = SafeHERDataset(
            records=self.manifest,
            root_dir=self.root_dir,
            split=DatasetSplit.TRAIN,
            target_workload=self.config.target_workload,
            in_memory_images=self.in_memory_images,
        )

        self.val_dataset = SafeHERDataset(
            records=self.manifest,
            root_dir=self.root_dir,
            split=DatasetSplit.VAL,
            target_workload=self.config.target_workload,
            in_memory_images=self.in_memory_images,
        )

        self.train_loader = create_safe_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.val_loader = create_safe_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # 5. Instantiate Model via Architecture Factory
        self.model = create_image_detector(
            architecture=self.config.architecture,
            model_version=self.config.model_version,
            is_trained=False,
        ).to(self.device)

        # 6. Loss criterion (CrossEntropyLoss with label smoothing)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        # 7. Optimizer (AdamW with decoupled weight decay)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 8. Learning Rate Scheduler
        self.scheduler: Optional[Any] = None
        if self.config.lr_scheduler_type == LRSchedulerType.COSINE:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        elif self.config.lr_scheduler_type == LRSchedulerType.PLATEAU:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_gamma,
                patience=2,
            )
        elif self.config.lr_scheduler_type == LRSchedulerType.STEP:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )

        # 9. Early Stopping Monitor
        es_mode = "min" if self.config.early_stopping_metric == EarlyStoppingMetric.VAL_LOSS else "max"
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode=es_mode,
        )

        # 10. Checkpoint Manager and Structured Logger
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            architecture=self.config.architecture,
            save_best_only=self.config.save_best_only,
        )

        self.logger = StructuredTrainingLogger(
            log_dir=self.config.log_dir,
            run_name=self.config.architecture,
        )

    def train_one_epoch(self) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        tp = 0
        fp = 0
        fn = 0

        for batch in self.train_loader:
            if batch is None or "images" not in batch:
                continue

            images = batch["images"].to(self.device)
            targets = batch["labels"].to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()

            tp += ((preds == 1) & (targets == 1)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "samples": total_samples,
        }

    def validate_one_epoch(self) -> Dict[str, float]:
        """Execute validation epoch over validation dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None or "images" not in batch:
                    continue

                images = batch["images"].to(self.device)
                targets = batch["labels"].to(self.device)
                batch_size = images.size(0)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()

                tp += ((preds == 1) & (targets == 1)).sum().item()
                fp += ((preds == 1) & (targets == 0)).sum().item()
                fn += ((preds == 0) & (targets == 1)).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "samples": total_samples,
        }

    def run(
        self,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ) -> TrainingPipelineResult:
        """Execute full training pipeline with checkpointing, logging, and early stopping.

        Args:
            resume_from_checkpoint: Optional path to checkpoint .pt file to resume training from.

        Returns:
            TrainingPipelineResult summary contract.
        """
        start_time = time.perf_counter()
        start_epoch = 0

        # Handle resume
        if resume_from_checkpoint is not None:
            ckpt_meta = self.checkpoint_manager.load_checkpoint_for_resume(
                checkpoint_path=resume_from_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                expected_manifest_hash=self.safety_audit.training_manifest_hash,
            )
            start_epoch = ckpt_meta.get("epoch", 0)
            logger.info(f"Resumed training from epoch {start_epoch} at {resume_from_checkpoint}")

        early_stopped = False
        completion_status = "COMPLETED"

        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            epoch_t0 = time.perf_counter()

            # Execute training & validation loops
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate_one_epoch()
            if train_metrics.get("samples", 0) <= 0:
                raise RuntimeError("Training split produced zero usable samples; refusing to create a trained model.")
            if val_metrics.get("samples", 0) <= 0:
                raise RuntimeError("Validation split produced zero usable samples; refusing to approve a checkpoint.")

            # Step scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Early stopping evaluation
            eval_score = (
                val_metrics["loss"]
                if self.config.early_stopping_metric == EarlyStoppingMetric.VAL_LOSS
                else val_metrics["accuracy"]
            )
            is_best = self.early_stopping.step(eval_score, epoch)

            epoch_dur = time.perf_counter() - epoch_t0

            # Structured JSONL logging
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                train_precision=train_metrics["precision"],
                train_recall=train_metrics["recall"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                learning_rate=current_lr,
                epoch_duration_seconds=epoch_dur,
                is_best=is_best,
                early_stopping_counter=self.early_stopping.counter,
            )

            # Persist checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                epoch=epoch,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                manifest_hash=self.safety_audit.training_manifest_hash,
                split_hash=self.safety_audit.split_manifest_hash,
                dataset_versions=self.safety_audit.dataset_versions,
                seed=self.config.seed,
                is_best=is_best,
                training_completed=False,
                approved_usage_state=(
                    "COMMERCIAL_TRAINED"
                    if self.safety_audit.commercial_approved
                    else "RESEARCH_RESTRICTED"
                ),
            )

            # Check early stopping condition
            if self.early_stopping.early_stop:
                logger.warning(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"monitored metric '{self.config.early_stopping_metric.value}' plateaued."
                )
                early_stopped = True
                completion_status = "EARLY_STOPPED"
                break

        total_duration = time.perf_counter() - start_time

        # Finalize only a real, non-empty training run with an approved dataset
        # and a validation result. Intermediate checkpoints remain non-deployable.
        if not self.logger.history:
            raise RuntimeError("No completed training epochs were recorded; refusing to mark model as trained.")
        self.model.is_trained = True
        final_checkpoint, final_checkpoint_sha256 = self.checkpoint_manager.finalize_best_checkpoint(self.model)
        best_ckpt_path = str(final_checkpoint)
        best_ckpt_sha256 = final_checkpoint_sha256

        # Exclusions count from dataset loaders
        total_exclusions = len(self.train_dataset.exclusion_log) + len(self.val_dataset.exclusion_log)

        result = TrainingPipelineResult(
            model_architecture=self.config.architecture,
            model_version=self.config.model_version,
            total_epochs_trained=len(self.logger.history),
            best_epoch=self.early_stopping.best_epoch or 1,
            best_val_loss=self.early_stopping.best_score if self.config.early_stopping_metric == EarlyStoppingMetric.VAL_LOSS else 0.0,
            best_val_accuracy=self.early_stopping.best_score if self.config.early_stopping_metric == EarlyStoppingMetric.VAL_ACCURACY else 0.0,
            early_stopped=early_stopped,
            best_checkpoint_path=best_ckpt_path,
            best_checkpoint_sha256=best_ckpt_sha256,
            training_manifest_hash=self.safety_audit.training_manifest_hash,
            split_manifest_hash=self.safety_audit.split_manifest_hash,
            dataset_versions=self.safety_audit.dataset_versions,
            history=self.logger.history,
            completion_status=completion_status,
            total_duration_seconds=round(total_duration, 2),
            device_used=self.device.type,
            exclusions_count=total_exclusions,
        )

        self.logger.save_summary(result.model_dump())
        return result
