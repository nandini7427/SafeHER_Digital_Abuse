"""Configuration schemas and result contracts for SafeHER neural training pipelines."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_core.datasets.manifest import TargetUsageScope
from ai_core.neural.schemas import MODEL_A_CLASS_MAPPING, NeuralArchitecture


class LRSchedulerType(str, Enum):
    """Supported learning-rate scheduler strategies."""

    COSINE = "cosine"
    PLATEAU = "plateau"
    STEP = "step"
    NONE = "none"


class EarlyStoppingMetric(str, Enum):
    """Metric evaluated for early-stopping triggering."""

    VAL_LOSS = "val_loss"
    VAL_ACCURACY = "val_accuracy"


class TrainingPipelineConfig(BaseModel):
    """Rigid, reproducible configuration contract for neural model training."""

    model_config = ConfigDict(frozen=True)

    architecture: str = Field(
        default=NeuralArchitecture.MOBILENET_V3_LARGE.value,
        description="Neural backbone identifier ('mobilenet_v3_large' or 'efficientnet_b0').",
    )
    model_version: str = Field(
        default="1.0.0-m4.4",
        description="Target semantic version string for trained model artifact.",
    )
    target_workload: TargetUsageScope = Field(
        default=TargetUsageScope.PRODUCTION_TRAINING,
        description="Target workload scope for legal and licensing validation.",
    )
    epochs: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Total training epochs.",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Per-batch sample count (bounded for CPU RAM limits).",
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        le=1.0,
        description="Initial learning rate for AdamW optimizer.",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        description="L2 regularization coefficient.",
    )
    label_smoothing: float = Field(
        default=0.05,
        ge=0.0,
        lt=0.5,
        description="Label smoothing factor applied in cross-entropy loss.",
    )
    lr_scheduler_type: LRSchedulerType = Field(
        default=LRSchedulerType.COSINE,
        description="Learning rate scheduler policy.",
    )
    lr_step_size: int = Field(
        default=5,
        ge=1,
        description="Step size for StepLR scheduler.",
    )
    lr_gamma: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Multiplicative decay factor for StepLR or Plateau schedulers.",
    )
    early_stopping_patience: int = Field(
        default=3,
        ge=1,
        description="Epochs to wait with no improvement before early stopping triggers.",
    )
    early_stopping_min_delta: float = Field(
        default=1e-4,
        ge=0.0,
        description="Minimum absolute delta required to qualify as improvement.",
    )
    early_stopping_metric: EarlyStoppingMetric = Field(
        default=EarlyStoppingMetric.VAL_LOSS,
        description="Validation metric monitored by early stopping.",
    )
    seed: int = Field(
        default=42,
        description="Master random seed for complete pipeline reproducibility.",
    )
    num_threads: int = Field(
        default=2,
        ge=1,
        le=16,
        description="CPU execution threads allocated to PyTorch.",
    )
    device: str = Field(
        default="auto",
        description="Computing device selection ('auto', 'cpu', or 'cuda').",
    )
    use_mixed_precision: bool = Field(
        default=False,
        description="Whether to use torch.cuda.amp mixed precision (disabled on CPU).",
    )
    gradient_clip_val: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for gradient clipping.",
    )
    checkpoint_dir: str = Field(
        default="checkpoints/model_a",
        description="Directory for persisting intermediate and best checkpoints.",
    )
    log_dir: str = Field(
        default="logs/model_a",
        description="Directory for structured JSONL training logs and summaries.",
    )
    save_best_only: bool = Field(
        default=True,
        description="Whether to retain only the best checkpoint by validation metric.",
    )
    enforce_approved_manifest: bool = Field(
        default=True,
        description="Enforces rights and leakage validation on dataset manifests before training.",
    )
    root_dir: Optional[str] = Field(
        default=None,
        description="Filesystem root path for resolving manifest image paths.",
    )


class TrainingPipelineResult(BaseModel):
    """Immutable result contract summarizing the outcome of a training pipeline run."""

    model_config = ConfigDict(frozen=True)

    model_architecture: str = Field(
        ...,
        description="Backbone architecture identifier.",
    )
    model_version: str = Field(
        ...,
        description="Trained model version string.",
    )
    total_epochs_trained: int = Field(
        ...,
        description="Total epochs completed during training.",
    )
    best_epoch: int = Field(
        ...,
        description="Epoch index where best validation metric was achieved.",
    )
    best_val_loss: float = Field(
        ...,
        description="Lowest validation loss achieved across training.",
    )
    best_val_accuracy: float = Field(
        ...,
        description="Validation accuracy corresponding to best checkpoint.",
    )
    early_stopped: bool = Field(
        default=False,
        description="Whether training terminated early due to metric plateau.",
    )
    best_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to best saved model weights checkpoint.",
    )
    best_checkpoint_sha256: Optional[str] = Field(
        default=None,
        description="Cryptographic SHA-256 digest of best checkpoint.",
    )
    training_manifest_hash: str = Field(
        ...,
        description="SHA-256 hash of training dataset manifest records.",
    )
    split_manifest_hash: str = Field(
        ...,
        description="SHA-256 hash of split allocation manifest.",
    )
    dataset_versions: List[str] = Field(
        default_factory=list,
        description="List of dataset versions incorporated in training.",
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Epoch-by-epoch quantitative training metrics history.",
    )
    completion_status: str = Field(
        ...,
        description="Final training run completion status ('COMPLETED', 'EARLY_STOPPED', 'RESUMED').",
    )
    total_duration_seconds: float = Field(
        ...,
        description="Total wall-clock duration of training in seconds.",
    )
    device_used: str = Field(
        ...,
        description="Device on which training was executed ('cpu' or 'cuda').",
    )
    exclusions_count: int = Field(
        default=0,
        description="Total count of corrupted/ineligible records excluded and audited.",
    )
