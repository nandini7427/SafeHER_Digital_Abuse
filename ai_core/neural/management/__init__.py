"""SafeHER Secure Model Artifact and Inference Management Framework."""

from ai_core.neural.management.artifacts import (
    ArchitectureMismatchError,
    ArtifactPathTraversalError,
    ArtifactSecurityPolicy,
    CorruptArtifactError,
    IntegrityVerificationError,
    SecureArtifactManager,
    UnsafeFormatError,
    compute_streaming_sha256,
)
from ai_core.neural.management.engine import (
    BatchLimitExceededError,
    EngineLifecycleState,
    InferenceTimeoutError,
    ModelVersionMismatchError,
    SecureEngineConfig,
    SecureInferenceEngine,
)
from ai_core.neural.management.metadata import (
    ModelArtifactFormat,
    ModelUsageState,
    SecureModelMetadata,
)
from ai_core.neural.management.recovery import (
    CandidateModelDeployment,
    DeploymentActivationError,
    DeploymentTrigger,
    ModelRollbackRecoveryController,
    ModelVersionTransition,
    RollbackUnavailableError,
    TransitionStatus,
    UnapprovedUsageStateError,
)

__all__ = [
    # Metadata
    "ModelUsageState",
    "ModelArtifactFormat",
    "SecureModelMetadata",
    # Artifact Security
    "ArtifactPathTraversalError",
    "IntegrityVerificationError",
    "CorruptArtifactError",
    "ArchitectureMismatchError",
    "UnsafeFormatError",
    "ArtifactSecurityPolicy",
    "SecureArtifactManager",
    "compute_streaming_sha256",
    # Inference Engine
    "EngineLifecycleState",
    "InferenceTimeoutError",
    "BatchLimitExceededError",
    "ModelVersionMismatchError",
    "SecureEngineConfig",
    "SecureInferenceEngine",
    # Model Rollback & Recovery
    "DeploymentTrigger",
    "TransitionStatus",
    "UnapprovedUsageStateError",
    "DeploymentActivationError",
    "RollbackUnavailableError",
    "ModelVersionTransition",
    "CandidateModelDeployment",
    "ModelRollbackRecoveryController",
]
