"""Checkpoint management, safe serialization, and training resume subsystem.

Persists full training state (model weights, optimizer state, scheduler state,
early-stopping state, and environmental/manifest metadata) with cryptographic SHA-256
verification and safe tensor serialization.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.factory import compute_file_sha256
from ai_core.neural.training.early_stopping import EarlyStopping


def get_training_environment() -> Dict[str, Any]:
    """Capture current system and runtime environment metadata."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_num_threads": torch.get_num_threads(),
    }


class CheckpointManager:
    """Manages saving, verification, and restoration of training checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        architecture: str,
        save_best_only: bool = True,
    ):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored.
            architecture: Architecture identifier.
            save_best_only: If True, only retains the best and latest checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.architecture = architecture
        self.save_best_only = save_best_only

        self.latest_checkpoint_path: Optional[Path] = None
        self.best_checkpoint_path: Optional[Path] = None
        self.best_checkpoint_sha256: Optional[str] = None

    def save_checkpoint(
        self,
        model: BaseNeuralDetector,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        early_stopping: EarlyStopping,
        epoch: int,
        val_loss: float,
        val_accuracy: float,
        manifest_hash: str,
        split_hash: str,
        dataset_versions: List[str],
        seed: int,
        is_best: bool = False,
        training_completed: bool = False,
        approved_usage_state: str = "PENDING_REVIEW",
    ) -> Tuple[Path, str]:
        """Save a complete training checkpoint payload.

        Args:
            model: Current detector model.
            optimizer: Current optimizer.
            scheduler: Optional learning rate scheduler.
            early_stopping: Current EarlyStopping monitor.
            epoch: Completed epoch index.
            val_loss: Validation loss for this epoch.
            val_accuracy: Validation accuracy for this epoch.
            manifest_hash: SHA-256 of training dataset manifest.
            split_hash: SHA-256 of split allocation manifest.
            dataset_versions: List of dataset versions.
            seed: Pipeline random seed.
            is_best: Whether this checkpoint achieved the best validation score.
            training_completed: Whether the training run has concluded.

        Returns:
            Tuple of (checkpoint_path, sha256_hash).
        """
        metadata: Dict[str, Any] = {
            "architecture": self.architecture,
            "model_version": model.get_metadata().model_version,
            "model_config": model.get_metadata().model_dump(),
            "class_mapping": model.class_mapping,
            "training_manifest_hash": manifest_hash,
            "split_manifest_hash": split_hash,
            "dataset_versions": dataset_versions,
            "random_seed": seed,
            "training_environment": get_training_environment(),
            "epoch": epoch,
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_accuracy, 5),
            "is_best": is_best,
            "training_completed": bool(training_completed),
            "is_trained": bool(training_completed and model.is_trained),
            "approved_usage_state": approved_usage_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "early_stopping_state": early_stopping.state_dict(),
            "metadata": metadata,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.architecture}_latest.pt"
        torch.save(payload, str(latest_path))
        latest_sha256 = compute_file_sha256(latest_path)
        self.latest_checkpoint_path = latest_path

        primary_path = latest_path
        primary_sha256 = latest_sha256

        # If best, copy or save as best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{self.architecture}_best.pt"
            torch.save(payload, str(best_path))
            best_sha256 = compute_file_sha256(best_path)
            self.best_checkpoint_path = best_path
            self.best_checkpoint_sha256 = best_sha256
            primary_path = best_path
            primary_sha256 = best_sha256

            # Also export standalone SafeTensors weights if safetensors is available
            try:
                import safetensors.torch
                safetensors_path = self.checkpoint_dir / f"{self.architecture}_best.safetensors"
                str_meta = {
                    "architecture": self.architecture,
                    "model_version": model.get_metadata().model_version,
                    "training_manifest_hash": manifest_hash,
                    "split_manifest_hash": split_hash,
                    "dataset_versions": json.dumps(dataset_versions),
                    "epoch": str(epoch),
                    "training_completed": json.dumps(bool(training_completed)),
                    "is_trained": json.dumps(bool(training_completed and model.is_trained)),
                    "approved_usage_state": approved_usage_state,
                }
                safetensors.torch.save_file(model.state_dict(), str(safetensors_path), metadata=str_meta)
            except Exception:
                pass

        # If retaining all epochs
        if not self.save_best_only:
            epoch_path = self.checkpoint_dir / f"{self.architecture}_epoch_{epoch:03d}.pt"
            torch.save(payload, str(epoch_path))

        return primary_path, primary_sha256

    def finalize_best_checkpoint(self, model: BaseNeuralDetector) -> Tuple[Path, str]:
        """Mark the best checkpoint deployable only after a completed training run.

        The final artifact includes explicit training completion and dataset/split
        provenance. Intermediate epoch checkpoints remain untrusted for inference.
        """
        if self.best_checkpoint_path is None or not self.best_checkpoint_path.is_file():
            raise RuntimeError("No best checkpoint exists; training cannot be marked complete.")

        payload = torch.load(str(self.best_checkpoint_path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
            raise ValueError("Best checkpoint has an invalid or incomplete training payload.")

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Best checkpoint is missing training metadata.")
        required = ("training_manifest_hash", "split_manifest_hash", "dataset_versions", "epoch", "architecture")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"Best checkpoint is missing provenance fields: {', '.join(missing)}")
        if metadata.get("architecture") != self.architecture:
            raise ValueError("Best checkpoint architecture does not match this checkpoint manager.")

        metadata["training_completed"] = True
        metadata["is_trained"] = True
        metadata["finalized_at"] = datetime.now(timezone.utc).isoformat()
        payload["metadata"] = metadata
        torch.save(payload, str(self.best_checkpoint_path))

        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.is_trained = True

        try:
            import safetensors.torch
            safetensors_path = self.checkpoint_dir / f"{self.architecture}_best.safetensors"
            str_meta = {
                key: value if isinstance(value, str) else json.dumps(value)
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool, list, dict))
            }
            safetensors.torch.save_file(model.state_dict(), str(safetensors_path), metadata=str_meta)
            self.best_checkpoint_path = safetensors_path
        except ImportError:
            # A completed, weights_only-compatible .pt artifact remains available.
            pass

        digest = compute_file_sha256(self.best_checkpoint_path)
        self.best_checkpoint_sha256 = digest
        model.weights_sha256 = digest
        return self.best_checkpoint_path, digest

    def load_checkpoint_for_resume(
        self,
        checkpoint_path: Union[str, Path],
        model: BaseNeuralDetector,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        expected_manifest_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore training state from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            model: Detector instance to load weights into.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional LR scheduler to restore state into.
            early_stopping: Optional EarlyStopping monitor to restore.
            expected_manifest_hash: Optional expected manifest hash for sanity check.

        Returns:
            Dictionary of recovered metadata (epoch, best_scores, hashes, etc.).

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If architecture or manifest hashes mismatch.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Resume checkpoint payload must be a dictionary.")

        # Reconstruct the best-checkpoint pointer when resuming a run in a new
        # pipeline instance. The training manager is intentionally stateless
        # across processes, so the filesystem is the source of truth.
        best_path = self.checkpoint_dir / f"{self.architecture}_best.pt"
        if best_path.is_file():
            self.best_checkpoint_path = best_path
            self.best_checkpoint_sha256 = compute_file_sha256(best_path)
        self.latest_checkpoint_path = path

        metadata = payload.get("metadata", {})
        ckpt_arch = metadata.get("architecture")
        if ckpt_arch and ckpt_arch != self.architecture:
            raise ValueError(
                f"Architecture mismatch in checkpoint! Expected '{self.architecture}', got '{ckpt_arch}'"
            )

        ckpt_manifest_hash = metadata.get("training_manifest_hash")
        if expected_manifest_hash and ckpt_manifest_hash:
            if ckpt_manifest_hash != expected_manifest_hash:
                raise ValueError(
                    f"Manifest hash mismatch! Checkpoint was trained on manifest {ckpt_manifest_hash}, "
                    f"but current pipeline specifies {expected_manifest_hash}"
                )

        # Restore model
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.is_trained = True

        # Restore optimizer
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        # Restore early stopping
        if early_stopping is not None and payload.get("early_stopping_state") is not None:
            early_stopping.load_state_dict(payload["early_stopping_state"])

        return metadata
