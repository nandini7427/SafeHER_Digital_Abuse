"""SafeHER Milestone 4 Neural Training Subsystem."""

from ai_core.neural.training.checkpoint_manager import (
    CheckpointManager,
    get_training_environment,
)
from ai_core.neural.training.checkpointer import save_detector_checkpoint
from ai_core.neural.training.config import (
    EarlyStoppingMetric,
    LRSchedulerType,
    TrainingPipelineConfig,
    TrainingPipelineResult,
)
from ai_core.neural.training.early_stopping import EarlyStopping
from ai_core.neural.training.logger import StructuredTrainingLogger
from ai_core.neural.training.pipeline import (
    ModelATrainingPipeline,
    set_reproducible_seed,
)
from ai_core.neural.training.safety import (
    ManifestSafetyAudit,
    compute_records_hash,
    compute_split_allocation_hash,
    enforce_training_safety_and_rights,
)
from ai_core.neural.training.trainer import EpochMetrics, ModelATrainer

__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "EarlyStoppingMetric",
    "EpochMetrics",
    "LRSchedulerType",
    "ManifestSafetyAudit",
    "ModelATrainer",
    "ModelATrainingPipeline",
    "StructuredTrainingLogger",
    "TrainingPipelineConfig",
    "TrainingPipelineResult",
    "compute_records_hash",
    "compute_split_allocation_hash",
    "enforce_training_safety_and_rights",
    "get_training_environment",
    "save_detector_checkpoint",
    "set_reproducible_seed",
]
