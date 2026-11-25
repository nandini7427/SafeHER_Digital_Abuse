"""Safe model-version rollback, atomic deployment, and operational recovery for SafeHER (M4).

Guarantees:
1. Active model version identification and tracking.
2. Candidate artifact pre-validation before activation (checksum, format, architecture, metadata).
3. Preservation of previously approved model versions for instant failback.
4. Atomic configuration and engine switching with zero partial artifact activation.
5. Post-deployment checksum verification detecting disk corruption / bit rot.
6. Automated and manual rollback runbooks on loading or compatibility failure.
7. Audited version transition history with cryptographic hashes and failure diagnostics.
8. Anti-pattern enforcement: NEVER silently switch to unverified models, and NEVER
   automatically enable experimental or unapproved models.
9. Worker restart recovery and state persistence idempotence.
10. Full thread-safety and continuous inference availability during version transitions.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import torch

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
    EngineLifecycleState,
    ModelVersionMismatchError,
    SecureEngineConfig,
    SecureInferenceEngine,
)
from ai_core.neural.management.metadata import (
    ModelArtifactFormat,
    ModelUsageState,
    SecureModelMetadata,
)
from ai_core.neural.models.factory import create_image_detector
from ai_core.neural.schemas import (
    InferenceStatus,
    InputConfiguration,
    ModelInferenceResult,
)

logger = logging.getLogger("safeher.ai.recovery")


class DeploymentTrigger(str, Enum):
    """Categorical trigger initiating a deployment or transition."""

    INITIAL_BOOT = "INITIAL_BOOT"
    VERSION_UPGRADE = "VERSION_UPGRADE"
    MANUAL_DEPLOYMENT = "MANUAL_DEPLOYMENT"
    AUTOMATED_ROLLBACK = "AUTOMATED_ROLLBACK"
    MANUAL_ROLLBACK = "MANUAL_ROLLBACK"
    WORKER_RESTART = "WORKER_RESTART"


class TransitionStatus(str, Enum):
    """Execution outcome status of a version transition."""

    SUCCESS = "SUCCESS"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_ACTIVATION = "FAILED_ACTIVATION"
    FAILED_POST_DEPLOYMENT = "FAILED_POST_DEPLOYMENT"
    ROLLED_BACK = "ROLLED_BACK"


class UnapprovedUsageStateError(PermissionError):
    """Raised when an unapproved, experimental, or research-restricted model is queued for production."""
    pass


class DeploymentActivationError(RuntimeError):
    """Raised when model deployment fails during pre-activation or parameter loading."""
    pass


class RollbackUnavailableError(RuntimeError):
    """Raised when rollback is requested but no previously stable model or baseline is available."""
    pass


@dataclass(frozen=True)
class ModelVersionTransition:
    """Immutable audit record documenting a model version transition."""

    transition_id: str
    timestamp: datetime
    model_id: str
    from_version: Optional[str]
    to_version: str
    trigger: DeploymentTrigger
    status: TransitionStatus
    weights_sha256: str
    error_message: Optional[str] = None
    operator_or_system: str = "system"
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "trigger": self.trigger.value,
            "status": self.status.value,
            "weights_sha256": self.weights_sha256,
            "error_message": self.error_message,
            "operator_or_system": self.operator_or_system,
            "duration_ms": self.duration_ms,
        }


@dataclass
class CandidateModelDeployment:
    """Deployment specification for a candidate neural model version."""

    model_id: str
    target_version: str
    artifact_path: Union[str, Path]
    metadata_path: Union[str, Path]
    expected_architecture: str = "mobilenet_v3_large"
    expected_sha256: Optional[str] = None
    calibration_artifact_path: Optional[Union[str, Path]] = None
    companion_models: Dict[str, Tuple[Union[str, Path], Union[str, Path]]] = None

    def __post_init__(self) -> None:
        self.artifact_path = Path(self.artifact_path)
        self.metadata_path = Path(self.metadata_path)
        if self.calibration_artifact_path:
            self.calibration_artifact_path = Path(self.calibration_artifact_path)
        if self.companion_models is None:
            self.companion_models = {}


class ModelRollbackRecoveryController:
    """Thread-safe controller governing atomic model deployment, validation, and zero-downtime recovery."""

    def __init__(
        self,
        security_policy: Optional[ArtifactSecurityPolicy] = None,
        engine_config: Optional[SecureEngineConfig] = None,
        state_persistence_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.security_policy = security_policy or ArtifactSecurityPolicy()
        self.engine_config = engine_config or SecureEngineConfig()
        self.artifact_manager = SecureArtifactManager(policy=self.security_policy)
        self.state_persistence_path = Path(state_persistence_path) if state_persistence_path else None

        # Thread synchronization lock (re-entrant lock)
        self._lock = threading.RLock()

        # Active model state
        self._active_version: Optional[str] = None
        self._active_metadata: Optional[SecureModelMetadata] = None
        self._active_engine: Optional[SecureInferenceEngine] = None
        self._active_companion_engines: Dict[str, SecureInferenceEngine] = {}

        # Previously approved stable model state (preserved for rollback)
        self._previous_stable_version: Optional[str] = None
        self._previous_stable_metadata: Optional[SecureModelMetadata] = None
        self._previous_stable_engine: Optional[SecureInferenceEngine] = None
        self._previous_companion_engines: Dict[str, SecureInferenceEngine] = {}

        # Transition audit history
        self._transition_history: List[ModelVersionTransition] = []

    # --------------------------------------------------------------------------
    # Version Identification & Properties
    # --------------------------------------------------------------------------

    @property
    def active_version(self) -> Optional[str]:
        """Return the semantic version string of the currently active model."""
        with self._lock:
            return self._active_version

    @property
    def active_metadata(self) -> Optional[SecureModelMetadata]:
        """Return the cryptographically verified metadata descriptor of the active model."""
        with self._lock:
            return self._active_metadata

    @property
    def previous_stable_version(self) -> Optional[str]:
        """Return the version string of the preserved fallback model."""
        with self._lock:
            return self._previous_stable_version

    @property
    def transition_history(self) -> List[ModelVersionTransition]:
        """Return the immutable audit history of model transitions."""
        with self._lock:
            return list(self._transition_history)

    def is_ready(self) -> bool:
        """Check whether an active inference engine is loaded and operational."""
        with self._lock:
            return self._active_engine is not None and self._active_engine.status == InferenceStatus.READY

    # --------------------------------------------------------------------------
    # Pre-Validation Pipeline
    # --------------------------------------------------------------------------

    def validate_candidate(
        self,
        candidate: CandidateModelDeployment,
    ) -> SecureModelMetadata:
        """Thoroughly pre-validate candidate artifact before any activation attempts.

        Enforces:
        1. File existence and non-zero length.
        2. Cryptographic metadata integrity.
        3. Version alignment (rejects silent substitution).
        4. Usage state compliance (NEVER automatically enables experimental/unapproved models).
        5. Streaming SHA-256 and byte-size match against on-disk weights.
        6. Architecture compatibility.

        Returns:
            Validated SecureModelMetadata.

        Raises:
            FileNotFoundError, UnapprovedUsageStateError, IntegrityVerificationError,
            ArchitectureMismatchError, ModelVersionMismatchError.
        """
        # 1. Check file existence
        if not candidate.artifact_path.is_file():
            raise FileNotFoundError(f"Candidate model weights not found at: {candidate.artifact_path}")
        if not candidate.metadata_path.is_file():
            raise FileNotFoundError(f"Candidate model metadata not found at: {candidate.metadata_path}")

        # 2. Load and verify metadata descriptor
        metadata = SecureModelMetadata.load(candidate.metadata_path)

        # 3. Verify version matches deployment specification
        if metadata.version != candidate.target_version:
            raise ModelVersionMismatchError(
                f"Candidate metadata version '{metadata.version}' does not match "
                f"requested deployment target '{candidate.target_version}'. Silent substitution is rejected."
            )

        # 4. Enforce Usage State Policy: NEVER enable experimental/unapproved models automatically
        if metadata.approved_usage_state != ModelUsageState.COMMERCIAL_TRAINED:
            raise UnapprovedUsageStateError(
                f"Candidate model '{candidate.model_id}:{candidate.target_version}' cannot be deployed. "
                f"Approved usage state is '{metadata.approved_usage_state.value}'. "
                f"SafeHER operational policy strictly requires '{ModelUsageState.COMMERCIAL_TRAINED.value}'. "
                "Experimental, research-restricted, or pending models must never be enabled automatically."
            )

        # 5. Architecture validation
        if metadata.architecture != candidate.expected_architecture:
            raise ArchitectureMismatchError(
                f"Architecture mismatch for candidate: expected '{candidate.expected_architecture}', "
                f"metadata declares '{metadata.architecture}'."
            )

        # 6. Verify streaming SHA-256 and file size
        actual_sha256, actual_size = compute_streaming_sha256(candidate.artifact_path)
        if actual_sha256 != metadata.sha256.lower():
            raise IntegrityVerificationError(
                f"Candidate weights SHA-256 mismatch! Metadata declares '{metadata.sha256}', "
                f"computed on disk '{actual_sha256}'."
            )

        if candidate.expected_sha256 and actual_sha256 != candidate.expected_sha256.lower():
            raise IntegrityVerificationError(
                f"Candidate weights do not match expected deploy hash: expected '{candidate.expected_sha256}', "
                f"computed '{actual_sha256}'."
            )

        if metadata.expected_file_size_bytes != actual_size:
            raise IntegrityVerificationError(
                f"Candidate weights file size mismatch: metadata declares {metadata.expected_file_size_bytes} bytes, "
                f"disk file is {actual_size} bytes."
            )

        # 7. Pre-validate all companion models if specified (preventing partial activation)
        for comp_name, (comp_art, comp_meta) in candidate.companion_models.items():
            comp_art_p = Path(comp_art)
            comp_meta_p = Path(comp_meta)
            if not comp_art_p.is_file() or not comp_meta_p.is_file():
                raise FileNotFoundError(f"Companion model '{comp_name}' files missing: {comp_art_p}, {comp_meta_p}")
            m_comp = SecureModelMetadata.load(comp_meta_p)
            sha_comp, size_comp = compute_streaming_sha256(comp_art_p)
            if sha_comp != m_comp.sha256.lower() or size_comp != m_comp.expected_file_size_bytes:
                raise IntegrityVerificationError(f"Companion model '{comp_name}' failed checksum verification.")

        return metadata

    # --------------------------------------------------------------------------
    # Atomic Model Deployment Execution
    # --------------------------------------------------------------------------

    def deploy_candidate(
        self,
        candidate: CandidateModelDeployment,
        trigger: DeploymentTrigger = DeploymentTrigger.VERSION_UPGRADE,
        operator_or_system: str = "system",
    ) -> ModelVersionTransition:
        """Atomically deploy candidate model version with pre-validation and zero-downtime rollback.

        Steps:
        1. Pre-validate candidate artifacts (metadata, hashes, usage state, formats).
        2. Instantiate candidate engine in memory and execute a warmup forward pass.
        3. Pre-load companion models; if any fails, abort deployment with zero partial activation.
        4. Preserve currently active engine in previous_stable storage.
        5. Perform atomic pointer swap under synchronization lock.
        6. Post-deployment checksum verification on disk. If failed, automatically rollback.
        7. Persist active version state for worker restart recovery.
        8. Record transition in audit trail.
        """
        start_time = time.perf_counter()
        transition_id = str(uuid.uuid4())
        from_ver = self.active_version

        with self._lock:
            # Phase 1: Pre-Validation
            try:
                candidate_meta = self.validate_candidate(candidate)
            except Exception as val_err:
                dur_ms = (time.perf_counter() - start_time) * 1000.0
                trans = ModelVersionTransition(
                    transition_id=transition_id,
                    timestamp=datetime.now(timezone.utc),
                    model_id=candidate.model_id,
                    from_version=from_ver,
                    to_version=candidate.target_version,
                    trigger=trigger,
                    status=TransitionStatus.FAILED_VALIDATION,
                    weights_sha256=candidate.expected_sha256 or "unknown",
                    error_message=f"Pre-validation failed: {val_err}",
                    operator_or_system=operator_or_system,
                    duration_ms=dur_ms,
                )
                self._transition_history.append(trans)
                logger.error("Deployment pre-validation failed for %s: %s", candidate.target_version, val_err)
                raise

            # Phase 2: In-Memory Preparation & Warmup (Dry-Run)
            candidate_engine: Optional[SecureInferenceEngine] = None
            candidate_companions: Dict[str, SecureInferenceEngine] = {}

            try:
                # Prepare candidate engine in isolated instance
                candidate_engine = SecureInferenceEngine(
                    artifact_path=candidate.artifact_path,
                    metadata_path=candidate.metadata_path,
                    config=self.engine_config,
                    security_policy=self.security_policy,
                    lazy_load=False,
                )
                candidate_engine.warmup(sample_count=1)

                # Prepare companion models (preventing partial activation)
                for c_name, (c_art, c_meta) in candidate.companion_models.items():
                    c_eng = SecureInferenceEngine(
                        artifact_path=c_art,
                        metadata_path=c_meta,
                        config=self.engine_config,
                        security_policy=self.security_policy,
                        lazy_load=False,
                    )
                    c_eng.warmup(sample_count=1)
                    candidate_companions[c_name] = c_eng

            except Exception as act_err:
                dur_ms = (time.perf_counter() - start_time) * 1000.0
                trans = ModelVersionTransition(
                    transition_id=transition_id,
                    timestamp=datetime.now(timezone.utc),
                    model_id=candidate.model_id,
                    from_version=from_ver,
                    to_version=candidate.target_version,
                    trigger=trigger,
                    status=TransitionStatus.FAILED_ACTIVATION,
                    weights_sha256=candidate_meta.sha256,
                    error_message=f"Candidate activation failed: {act_err}",
                    operator_or_system=operator_or_system,
                    duration_ms=dur_ms,
                )
                self._transition_history.append(trans)
                logger.critical("Activation failed for %s; previous model remains active: %s", candidate.target_version, act_err)
                raise DeploymentActivationError(f"Candidate activation failed: {act_err}") from act_err

            # Phase 3: Preserve Previously Approved Model & Atomic Pointer Swap
            self._previous_stable_version = self._active_version
            self._previous_stable_metadata = self._active_metadata
            self._previous_stable_engine = self._active_engine
            self._previous_companion_engines = dict(self._active_companion_engines)

            # Atomic swap
            self._active_engine = candidate_engine
            self._active_metadata = candidate_meta
            self._active_version = candidate.target_version
            self._active_companion_engines = candidate_companions

            # Phase 4: Post-Deployment Checksum Verification
            # Verifies that disk file wasn't corrupted or swapped concurrently
            post_sha, post_size = compute_streaming_sha256(candidate.artifact_path)
            if post_sha != candidate_meta.sha256.lower() or post_size != candidate_meta.expected_file_size_bytes:
                logger.critical(
                    "SECURITY HAZARD: Post-deployment checksum verification failed for %s! Initiating automatic rollback.",
                    candidate.target_version,
                )
                # Immediate automatic rollback
                self._execute_internal_rollback(
                    reason="Post-deployment checksum mismatch (possible file corruption)",
                    trigger=DeploymentTrigger.AUTOMATED_ROLLBACK,
                    operator_or_system="system_post_verify_guard",
                )
                dur_ms = (time.perf_counter() - start_time) * 1000.0
                trans = ModelVersionTransition(
                    transition_id=transition_id,
                    timestamp=datetime.now(timezone.utc),
                    model_id=candidate.model_id,
                    from_version=from_ver,
                    to_version=candidate.target_version,
                    trigger=trigger,
                    status=TransitionStatus.FAILED_POST_DEPLOYMENT,
                    weights_sha256=post_sha,
                    error_message="Post-deployment checksum mismatch detected; rolled back.",
                    operator_or_system=operator_or_system,
                    duration_ms=dur_ms,
                )
                self._transition_history.append(trans)
                raise IntegrityVerificationError("Post-deployment checksum mismatch; rolled back to previous model.")

            # Phase 5: Persist Active State (for worker restart recovery)
            if self.state_persistence_path:
                self._save_state_file(candidate, candidate_meta)

            # Record successful transition
            dur_ms = (time.perf_counter() - start_time) * 1000.0
            trans = ModelVersionTransition(
                transition_id=transition_id,
                timestamp=datetime.now(timezone.utc),
                model_id=candidate.model_id,
                from_version=from_ver,
                to_version=candidate.target_version,
                trigger=trigger,
                status=TransitionStatus.SUCCESS,
                weights_sha256=candidate_meta.sha256,
                error_message=None,
                operator_or_system=operator_or_system,
                duration_ms=dur_ms,
            )
            self._transition_history.append(trans)
            logger.info("Successfully deployed model %s (v%s) in %.2fms", candidate.model_id, candidate.target_version, dur_ms)
            return trans

    # --------------------------------------------------------------------------
    # Rollback Execution Runbook
    # --------------------------------------------------------------------------

    def rollback(
        self,
        reason: str = "Operator requested rollback",
        operator_or_system: str = "operator",
    ) -> ModelVersionTransition:
        """Execute deterministic zero-downtime rollback to the preserved previously approved model."""
        with self._lock:
            return self._execute_internal_rollback(
                reason=reason,
                trigger=DeploymentTrigger.MANUAL_ROLLBACK,
                operator_or_system=operator_or_system,
            )

    def _execute_internal_rollback(
        self,
        reason: str,
        trigger: DeploymentTrigger,
        operator_or_system: str,
    ) -> ModelVersionTransition:
        """Internal synchronized rollback execution logic."""
        start_time = time.perf_counter()
        transition_id = str(uuid.uuid4())
        current_ver = self._active_version

        if not self._previous_stable_engine or not self._previous_stable_version:
            logger.error("Rollback failed: No previous stable model registered in memory.")
            raise RollbackUnavailableError("Cannot roll back: no previous stable model version is registered.")

        target_ver = self._previous_stable_version
        target_meta = self._previous_stable_metadata

        # Swap active and previous
        self._active_engine = self._previous_stable_engine
        self._active_version = self._previous_stable_version
        self._active_metadata = self._previous_stable_metadata
        self._active_companion_engines = dict(self._previous_companion_engines)

        # Clear previous stable to prevent circular rollback loops
        self._previous_stable_engine = None
        self._previous_stable_version = None
        self._previous_stable_metadata = None
        self._previous_companion_engines = {}

        # Update persistent state file if configured
        if self.state_persistence_path:
            try:
                state_data = {
                    "model_id": target_meta.model_id if target_meta else "model_a",
                    "version": target_ver,
                    "artifact_path": str(self._active_engine.artifact_path.resolve()) if self._active_engine else "",
                    "metadata_path": str(self._active_engine.metadata_path.resolve()) if self._active_engine else "",
                    "expected_architecture": target_meta.architecture if target_meta else "mobilenet_v3_large",
                    "expected_sha256": target_meta.sha256 if target_meta else "",
                    "rollback_incident_timestamp": datetime.now(timezone.utc).isoformat(),
                    "rollback_reason": reason,
                }
                with open(self.state_persistence_path, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2)
            except Exception as e:
                logger.warning("Failed to update persistent state file on rollback: %s", e)

        dur_ms = (time.perf_counter() - start_time) * 1000.0
        trans = ModelVersionTransition(
            transition_id=transition_id,
            timestamp=datetime.now(timezone.utc),
            model_id=target_meta.model_id if target_meta else "model",
            from_version=current_ver,
            to_version=target_ver,
            trigger=trigger,
            status=TransitionStatus.ROLLED_BACK,
            weights_sha256=target_meta.sha256 if target_meta else "unknown",
            error_message=f"Rollback executed: {reason}",
            operator_or_system=operator_or_system,
            duration_ms=dur_ms,
        )
        self._transition_history.append(trans)
        logger.warning("Rolled back model from %s to %s (Reason: %s)", current_ver, target_ver, reason)
        return trans

    # --------------------------------------------------------------------------
    # Worker Restart & State Persistence Idempotence
    # --------------------------------------------------------------------------

    def _save_state_file(
        self,
        candidate: CandidateModelDeployment,
        metadata: SecureModelMetadata,
    ) -> None:
        """Save active model specification to persistent state file."""
        if not self.state_persistence_path:
            return
        self.state_persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": candidate.model_id,
            "version": candidate.target_version,
            "artifact_path": str(candidate.artifact_path.resolve()),
            "metadata_path": str(candidate.metadata_path.resolve()),
            "expected_architecture": candidate.expected_architecture,
            "expected_sha256": metadata.sha256,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.state_persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def recover_on_worker_restart(
        self,
        fallback_candidate: Optional[CandidateModelDeployment] = None,
    ) -> Optional[ModelVersionTransition]:
        """Restore active model state on worker process startup or recovery.

        Steps:
        1. Read persisted state file if present.
        2. Pre-validate on-disk artifact integrity.
        3. If state file is corrupted or missing, fallback cleanly to fallback_candidate.
        4. If all fail, remain in safe uninitialized state without crashing the host worker.
        """
        with self._lock:
            target_candidate = None

            if self.state_persistence_path and self.state_persistence_path.is_file():
                try:
                    with open(self.state_persistence_path, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    target_candidate = CandidateModelDeployment(
                        model_id=saved["model_id"],
                        target_version=saved["version"],
                        artifact_path=saved["artifact_path"],
                        metadata_path=saved["metadata_path"],
                        expected_architecture=saved.get("expected_architecture", "mobilenet_v3_large"),
                        expected_sha256=saved.get("expected_sha256"),
                    )
                except Exception as parse_err:
                    logger.warning("Corrupted worker state file '%s': %s. Using fallback.", self.state_persistence_path, parse_err)
                    target_candidate = fallback_candidate
            else:
                target_candidate = fallback_candidate

            if target_candidate is None:
                logger.info("No saved deployment state or fallback candidate provided. Engine remains uninitialized.")
                return None

            try:
                return self.deploy_candidate(
                    candidate=target_candidate,
                    trigger=DeploymentTrigger.WORKER_RESTART,
                    operator_or_system="worker_startup_daemon",
                )
            except Exception as rec_err:
                logger.critical("Worker recovery failed for candidate %s: %s", target_candidate.target_version, rec_err)
                if fallback_candidate is not None and target_candidate != fallback_candidate:
                    logger.info("Attempting recovery with fallback baseline candidate %s...", fallback_candidate.target_version)
                    try:
                        return self.deploy_candidate(
                            candidate=fallback_candidate,
                            trigger=DeploymentTrigger.WORKER_RESTART,
                            operator_or_system="worker_fallback_recovery",
                        )
                    except Exception as fb_err:
                        logger.critical("Fallback candidate deployment also failed: %s", fb_err)
                return None

    # --------------------------------------------------------------------------
    # Concurrent Inference Execution
    # --------------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> ModelInferenceResult:
        """Execute thread-safe inference on the currently active model engine.

        Guarantees:
        - Inferences running during an atomic deployment finish smoothly on their engine.
        - New inferences immediately route to the newly swapped engine.
        """
        with self._lock:
            engine = self._active_engine

        if engine is None:
            return ModelInferenceResult(
                raw_logits=[],
                uncalibrated_scores={},
                predicted_class="UNKNOWN",
                model_version=self._active_version or "NONE",
                architecture="NONE",
                input_config=InputConfiguration(),
                status=InferenceStatus.MODEL_UNAVAILABLE,
                latency_ms=0.0,
                disclaimer="Model engine is uninitialized or undergoing recovery.",
            )

        return engine.predict(x)
