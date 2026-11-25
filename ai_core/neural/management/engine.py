"""Secure neural inference execution engine with lifecycle management, batch limits, and thread safety.

Features:
- Lazy model loading: Does not instantiate or deserialize heavy weights until explicitly requested.
- No auto-download: Loads strictly from local, verified filesystem paths; never dials external servers at startup.
- Device selection: Deterministic CPU execution by default with explicit GPU/MPS opt-in.
- Input validation: Rigorous shape, channel count, and non-finite value checks.
- Batch limits: Enforces max batch size to protect against memory exhaustion DoS.
- Memory-aware execution: Evaluates within torch.no_grad() in eval mode with garbage cleanup.
- Inference timeout behaviour: Monitors CPU execution and enforces wall-clock timeouts.
- Safe unavailable states: Returns typed non-fabricated triage results when artifacts are missing or invalid.
- Lifecycle handling: UNINITIALIZED -> LOADING -> READY -> UNLOADED -> SHUTDOWN.
- Thread-safety: Thread-synchronized execution supporting high-concurrency environments.
- Version protection: Strictly rejects silent substitution of differing model versions.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
import gc
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from ai_core.neural.management.artifacts import (
    ArchitectureMismatchError,
    ArtifactPathTraversalError,
    ArtifactSecurityPolicy,
    CorruptArtifactError,
    IntegrityVerificationError,
    SecureArtifactManager,
    UnsafeFormatError,
)
from ai_core.neural.management.metadata import (
    ModelUsageState,
    SecureModelMetadata,
)
from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.factory import create_image_detector
from ai_core.neural.schemas import (
    MANDATORY_NEURAL_DISCLAIMER,
    BatchInferenceResult,
    InferenceStatus,
    InputConfiguration,
    ModelInferenceResult,
)


class EngineLifecycleState(str, Enum):
    """Operational lifecycle state of the secure inference engine."""

    UNINITIALIZED = "UNINITIALIZED"
    LOADING = "LOADING"
    READY = "READY"
    BUSY = "BUSY"
    UNLOADED = "UNLOADED"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class InferenceTimeoutError(TimeoutError):
    """Raised when inference execution exceeds maximum allotted wall-clock duration."""
    pass


class BatchLimitExceededError(ValueError):
    """Raised when input batch exceeds the configured maximum batch size."""
    pass


class ModelVersionMismatchError(ValueError):
    """Raised when the loaded model version does not match the requested target version."""
    pass


@dataclass(frozen=True)
class SecureEngineConfig:
    """Configuration for the SecureInferenceEngine."""

    device: str = "cpu"
    num_threads: int = 2
    max_batch_size: int = 16
    allow_batch_chunking: bool = True
    timeout_seconds: float = 10.0
    deterministic: bool = True
    seed: int = 42


class SecureInferenceEngine:
    """Thread-safe, memory-bounded neural inference engine for SafeHER image models."""

    def __init__(
        self,
        artifact_path: Union[str, Path],
        metadata_path: Union[str, Path],
        config: Optional[SecureEngineConfig] = None,
        security_policy: Optional[ArtifactSecurityPolicy] = None,
        lazy_load: bool = True,
    ) -> None:
        self.config = config or SecureEngineConfig()
        self.security_policy = security_policy or ArtifactSecurityPolicy()
        self.artifact_manager = SecureArtifactManager(policy=self.security_policy)

        self.artifact_path = Path(artifact_path)
        self.metadata_path = Path(metadata_path)

        # Thread synchronization lock
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="safeher-infer")

        # Lifecycle & model state
        self._lifecycle_state = EngineLifecycleState.UNINITIALIZED
        self._status = InferenceStatus.READY
        self._status_message = "Uninitialized"
        self._model: Optional[BaseNeuralDetector] = None
        self._metadata: Optional[SecureModelMetadata] = None

        # Configure deterministic CPU execution
        if self.config.device == "cpu":
            torch.set_num_threads(max(1, self.config.num_threads))
            if self.config.deterministic:
                torch.manual_seed(self.config.seed)

        if not lazy_load:
            self.load()

    @property
    def lifecycle_state(self) -> EngineLifecycleState:
        with self._lock:
            return self._lifecycle_state

    @property
    def status(self) -> InferenceStatus:
        with self._lock:
            return self._status

    @property
    def metadata(self) -> Optional[SecureModelMetadata]:
        with self._lock:
            return self._metadata

    def load(self, target_version: Optional[str] = None) -> None:
        """Load model metadata and weights under lock with full verification.

        Args:
            target_version: Optional expected version. If specified, strictly verifies match
                           and refuses silent substitution of a different version.
        """
        with self._lock:
            if self._lifecycle_state in (EngineLifecycleState.READY, EngineLifecycleState.BUSY):
                if target_version and self._metadata and self._metadata.version != target_version:
                    raise ModelVersionMismatchError(
                        f"Active model version '{self._metadata.version}' does not match "
                        f"requested target version '{target_version}'. Silent substitution is rejected."
                    )
                return

            self._lifecycle_state = EngineLifecycleState.LOADING

            try:
                # 1. Load and verify metadata descriptor
                self._metadata = SecureModelMetadata.load(self.metadata_path)

                # 2. Verify target version if requested
                if target_version and self._metadata.version != target_version:
                    raise ModelVersionMismatchError(
                        f"Model metadata version '{self._metadata.version}' does not match "
                        f"requested version '{target_version}'. Silent substitution is rejected."
                    )

                # 3. Instantiate model architecture skeleton
                input_config = InputConfiguration(
                    spatial_resolution=(self._metadata.input_dimensions[1], self._metadata.input_dimensions[2]),
                    expected_channels=self._metadata.input_dimensions[0],
                )
                detector = create_image_detector(
                    architecture=self._metadata.architecture,
                    num_classes=len(self._metadata.class_mappings),
                    class_mapping=self._metadata.class_mappings,
                    input_config=input_config,
                    model_version=self._metadata.version,
                )

                # 4. Verify artifact checksum, format, and inject weights
                self._model = self.artifact_manager.load_and_verify_model(
                    artifact_path=self.artifact_path,
                    metadata=self._metadata,
                    model=detector,
                    device=self.config.device,
                )

                self._model.to(self.config.device)
                self._model.eval()

                self._lifecycle_state = EngineLifecycleState.READY
                self._status = InferenceStatus.READY
                self._status_message = "Model verified and loaded successfully."

            except (
                FileNotFoundError,
                ArtifactPathTraversalError,
                IntegrityVerificationError,
                CorruptArtifactError,
                ArchitectureMismatchError,
                UnsafeFormatError,
                ModelVersionMismatchError,
            ) as known_err:
                self._lifecycle_state = EngineLifecycleState.ERROR
                self._status = InferenceStatus.INTEGRITY_VERIFICATION_FAILED
                self._status_message = str(known_err)
                raise
            except Exception as ex:
                self._lifecycle_state = EngineLifecycleState.ERROR
                self._status = InferenceStatus.ERROR
                self._status_message = f"Unexpected loading failure: {ex}"
                raise

    def warmup(self, sample_count: int = 1) -> None:
        """Warm up model inference engine with synthetic input tensors."""
        with self._lock:
            if self._lifecycle_state != EngineLifecycleState.READY:
                self.load()

            c, h, w = self._metadata.input_dimensions if self._metadata else (3, 224, 224)
            dummy_tensor = torch.zeros((sample_count, c, h, w), dtype=torch.float32, device=self.config.device)
            with torch.no_grad():
                _ = self._model(dummy_tensor)

    def validate_tensor(self, x: torch.Tensor) -> None:
        """Rigorous input tensor shape, channel, dtype, and numerical sanity validation."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (Batch, Channels, Height, Width), got {x.ndim}D tensor."
            )

        b, c, h, w = x.shape
        if self._metadata is not None:
            expected_c, expected_h, expected_w = self._metadata.input_dimensions
        else:
            expected_c, expected_h, expected_w = 3, 224, 224

        if c != expected_c:
            raise ValueError(f"Input channels mismatch! Expected {expected_c}, got {c}.")
        if h != expected_h or w != expected_w:
            raise ValueError(
                f"Spatial resolution mismatch! Expected ({expected_h}, {expected_w}), got ({h}, {w})."
            )

        if not torch.all(torch.isfinite(x)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf).")

    def _execute_forward_with_timeout(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass with wall-clock timeout protection."""
        def forward_task() -> torch.Tensor:
            with torch.no_grad():
                return self._model(x.to(self.config.device))

        future = self._executor.submit(forward_task)
        try:
            return future.result(timeout=self.config.timeout_seconds)
        except FuturesTimeoutError:
            raise InferenceTimeoutError(
                f"Inference execution exceeded timeout threshold ({self.config.timeout_seconds}s)."
            )

    def predict(self, x: torch.Tensor) -> ModelInferenceResult:
        """Execute inference on a single sample or batch of shape (1, C, H, W)."""
        with self._lock:
            # Lazy loading if needed
            if self._lifecycle_state != EngineLifecycleState.READY:
                try:
                    self.load()
                except Exception as ex:
                    return self._create_unavailable_result(status=self._status, message=str(ex))

            # Input validation
            self.validate_tensor(x)

            if x.shape[0] != 1:
                # Delegate to predict_batch for multiple samples
                batch_res = self.predict_batch(x)
                return batch_res.results[0]

            self._lifecycle_state = EngineLifecycleState.BUSY
            start_time = time.perf_counter()

            try:
                logits_tensor = self._execute_forward_with_timeout(x)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                logits = logits_tensor[0].cpu().numpy().tolist()
                probs = torch.softmax(logits_tensor[0], dim=-1).cpu().numpy().tolist()

                class_mappings = self._metadata.class_mappings if self._metadata else {0: "CAMERA_PHOTOGRAPH", 1: "AI_GENERATED_IMAGE"}
                scores: Dict[str, float] = {}
                for idx, score in enumerate(probs):
                    label = class_mappings.get(idx, f"CLASS_{idx}")
                    scores[label] = float(score)

                pred_idx = int(torch.argmax(logits_tensor[0]).item())
                predicted_class = class_mappings.get(pred_idx, f"CLASS_{pred_idx}")

                return ModelInferenceResult(
                    raw_logits=[float(v) for v in logits],
                    uncalibrated_scores=scores,
                    predicted_class=predicted_class,
                    model_version=self._metadata.version if self._metadata else "unknown",
                    architecture=self._metadata.architecture if self._metadata else "unknown",
                    input_config=self._model.input_config if self._model else InputConfiguration(),
                    status=InferenceStatus.READY,
                    latency_ms=round(elapsed_ms, 3),
                    disclaimer=MANDATORY_NEURAL_DISCLAIMER,
                )
            finally:
                self._lifecycle_state = EngineLifecycleState.READY

    def predict_batch(self, x: torch.Tensor) -> BatchInferenceResult:
        """Execute batched inference with batch size limiting and chunking."""
        with self._lock:
            if self._lifecycle_state != EngineLifecycleState.READY:
                try:
                    self.load()
                except Exception as ex:
                    batch_len = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim >= 1 else 0
                    results = [self._create_unavailable_result(status=self._status, message=str(ex)) for _ in range(batch_len)]
                    return BatchInferenceResult(
                        results=results,
                        batch_size=batch_len,
                        total_latency_ms=0.0,
                        average_latency_ms=0.0,
                    )

            self.validate_tensor(x)
            batch_size = x.shape[0]

            if batch_size > self.config.max_batch_size:
                if not self.config.allow_batch_chunking:
                    raise BatchLimitExceededError(
                        f"Batch size ({batch_size}) exceeds maximum allowed limit "
                        f"({self.config.max_batch_size})."
                    )

            self._lifecycle_state = EngineLifecycleState.BUSY
            start_time = time.perf_counter()

            try:
                # Process in chunks of max_batch_size
                chunk_size = self.config.max_batch_size
                all_logits_list: List[torch.Tensor] = []

                for i in range(0, batch_size, chunk_size):
                    chunk = x[i : i + chunk_size]
                    chunk_logits = self._execute_forward_with_timeout(chunk)
                    all_logits_list.append(chunk_logits)

                full_logits = torch.cat(all_logits_list, dim=0)
                total_latency_ms = (time.perf_counter() - start_time) * 1000.0
                avg_latency_ms = total_latency_ms / max(1, batch_size)

                probs_tensor = torch.softmax(full_logits, dim=-1)
                class_mappings = self._metadata.class_mappings if self._metadata else {0: "CAMERA_PHOTOGRAPH", 1: "AI_GENERATED_IMAGE"}

                results: List[ModelInferenceResult] = []
                for i in range(batch_size):
                    sample_logits = full_logits[i].cpu().numpy().tolist()
                    sample_probs = probs_tensor[i].cpu().numpy().tolist()

                    scores: Dict[str, float] = {}
                    for idx, score in enumerate(sample_probs):
                        label = class_mappings.get(idx, f"CLASS_{idx}")
                        scores[label] = float(score)

                    pred_idx = int(torch.argmax(full_logits[i]).item())
                    pred_class = class_mappings.get(pred_idx, f"CLASS_{pred_idx}")

                    results.append(
                        ModelInferenceResult(
                            raw_logits=[float(v) for v in sample_logits],
                            uncalibrated_scores=scores,
                            predicted_class=pred_class,
                            model_version=self._metadata.version if self._metadata else "unknown",
                            architecture=self._metadata.architecture if self._metadata else "unknown",
                            input_config=self._model.input_config if self._model else InputConfiguration(),
                            status=InferenceStatus.READY,
                            latency_ms=round(avg_latency_ms, 3),
                            disclaimer=MANDATORY_NEURAL_DISCLAIMER,
                        )
                    )

                return BatchInferenceResult(
                    results=results,
                    batch_size=batch_size,
                    total_latency_ms=round(total_latency_ms, 3),
                    average_latency_ms=round(avg_latency_ms, 3),
                )
            finally:
                self._lifecycle_state = EngineLifecycleState.READY

    def _create_unavailable_result(self, status: InferenceStatus, message: str) -> ModelInferenceResult:
        """Create structured empty triage result for unavailable or error states."""
        return ModelInferenceResult(
            raw_logits=[],
            uncalibrated_scores={},
            predicted_class="UNKNOWN",
            model_version=self._metadata.version if self._metadata else "uninitialized",
            architecture=self._metadata.architecture if self._metadata else "unknown",
            input_config=InputConfiguration(),
            status=status,
            latency_ms=0.0,
            disclaimer=f"{MANDATORY_NEURAL_DISCLAIMER} (Engine Status: {status.value} - {message})",
        )

    def unload(self) -> None:
        """Unload model from RAM, release tensors, and trigger garbage collection."""
        with self._lock:
            self._model = None
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self._lifecycle_state = EngineLifecycleState.UNLOADED
            self._status = InferenceStatus.MODEL_UNAVAILABLE
            self._status_message = "Model explicitly unloaded."

    def shutdown(self) -> None:
        """Gracefully shut down inference engine and internal thread executors."""
        with self._lock:
            self.unload()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._lifecycle_state = EngineLifecycleState.SHUTDOWN
            self._status_message = "Inference engine shut down."
