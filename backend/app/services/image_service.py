"""Backend service layer for SafeHER Image Intelligence and Evidence Ingestion.

Handles secure streaming uploads, size limits, case ownership checks, non-blocking
inference execution, private evidence persistence, and privacy-safe audit logging.
"""

import asyncio
import concurrent.futures
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_core.image_intelligence.monitoring import model_monitoring_engine
from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.schemas import ImageAssessmentResponse
from ai_core.image_intelligence.service import image_service as ai_image_service
from ai_core.neural.facial.service import ModelBFacialInferenceService
from ai_core.neural.integration.pipeline import IntegratedForensicPipeline
from ai_core.neural.integration.schemas import (
    IntegratedForensicAssessment,
    NeuralInferenceMetadata,
)
from ai_core.neural.management.artifacts import (
    ArtifactSecurityPolicy,
    compute_streaming_sha256,
)
from ai_core.neural.management.engine import (
    InferenceTimeoutError,
    SecureEngineConfig,
    SecureInferenceEngine,
)
from backend.app.config import settings
from backend.app.models import AuditLog, Case, Evidence, EvidenceCleanupLease, User
from backend.app.services.file_locking import FileLockUnavailable, try_lock_exclusive, unlock_exclusive

logger = logging.getLogger("safeher.backend.services.image")

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


class _CapacityWaiter:
    __slots__ = ("loop", "future", "state")

    def __init__(self, loop: asyncio.AbstractEventLoop, future: asyncio.Future) -> None:
        self.loop = loop
        self.future = future
        self.state = "WAITING"  # WAITING, GRANTED, ACQUIRED, CANCELLED


class ProcessCapacityLimiter:
    """Thread-safe, event-loop-agnostic limiter with cancellation-safe FIFO handoff.

    A queued waiter that is cancelled at the same moment a permit is handed to it
    returns that permit to the queue. This prevents a rare but permanent capacity
    leak when cancellation races with ``call_soon_threadsafe``.
    """

    def __init__(self, max_permits: int) -> None:
        if type(max_permits) is not int or max_permits < 1:
            raise ValueError("max_permits must be a positive integer")
        self.max_permits = max_permits
        self._lock = threading.Lock()
        self._active_count = 0
        self._waiters: list[_CapacityWaiter] = []

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        waiter = _CapacityWaiter(loop, future)

        with self._lock:
            if self._active_count < self.max_permits and not self._waiters:
                self._active_count += 1
                return
            self._waiters.append(waiter)

        try:
            await future
        except BaseException:
            # Cancel a queued waiter, or return a permit that was granted but not
            # yet consumed by the coroutine. BaseException is intentional because
            # asyncio.CancelledError derives from BaseException.
            with self._lock:
                if waiter.state == "WAITING":
                    waiter.state = "CANCELLED"
                elif waiter.state == "GRANTED":
                    waiter.state = "CANCELLED"
                    self._release_internal_locked()
            raise
        else:
            with self._lock:
                if waiter.state != "GRANTED":
                    raise RuntimeError("Capacity waiter resumed without a granted permit")
                waiter.state = "ACQUIRED"

    def release(self) -> None:
        with self._lock:
            if self._active_count <= 0:
                raise RuntimeError("Capacity limiter released without an active permit")
            self._release_internal_locked()

    def _release_internal_locked(self) -> None:
        while self._waiters:
            waiter = self._waiters.pop(0)
            if waiter.state != "WAITING":
                continue
            if waiter.loop.is_closed():
                waiter.state = "CANCELLED"
                continue

            waiter.state = "GRANTED"
            try:
                waiter.loop.call_soon_threadsafe(self._complete_waiter, waiter)
            except RuntimeError:
                # The event loop closed between the is_closed check and scheduling.
                waiter.state = "CANCELLED"
                continue
            # Transfer the already-counted permit; active_count is unchanged.
            return

        self._active_count -= 1

    @staticmethod
    def _complete_waiter(waiter: _CapacityWaiter) -> None:
        if not waiter.future.done():
            waiter.future.set_result(None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    @property
    def available_permits(self) -> int:
        with self._lock:
            return max(0, self.max_permits - self._active_count)


class ContainerCapacityCoordinator:
    """Kernel-advisory-locked slot manager coordinating concurrency across all Uvicorn worker processes.

    Guarantees:
    - Kernel-enforced bounds across all worker processes using native OS advisory file locks on a shared lock directory.
    - Automatic release on worker crash/restart (OS kernel closes file handles and releases locks).
    - Cannot strand capacity slots across deployments or ungraceful terminations.
    - Real-time probe of actively running computations across the entire container/host.
    """

    def __init__(
        self,
        slots_dir: Optional[Path] = None,
        max_slots: Optional[int] = None,
    ) -> None:
        if max_slots is None:
            max_slots = int(
                os.environ.get(
                    "CONTAINER_MAX_CONCURRENT_ANALYSES",
                    str(settings.CONTAINER_MAX_CONCURRENT_ANALYSES),
                )
            )
        self.max_slots = max(1, max_slots)

        if slots_dir is None:
            base_dir = (
                Path(settings.CAPACITY_SLOTS_DIR)
                if settings.CAPACITY_SLOTS_DIR
                else (Path(settings.PRIVATE_STORAGE_PATH) / ".capacity_slots")
            )
            self.slots_dir = base_dir.resolve()
        else:
            self.slots_dir = Path(slots_dir).resolve()

        self.slots_dir.mkdir(parents=True, exist_ok=True)
        self._local_locks: dict[int, Any] = {}
        self._mutex = threading.Lock()

    def acquire_slot(self, timeout: float = 30.0, poll_interval: float = 0.02) -> Tuple[int, Any]:
        """Attempt to acquire an available slot lock across workers.

        Polls until a slot is acquired or timeout expires.
        Raises TimeoutError if unable to acquire within timeout.
        """
        deadline = time.perf_counter() + timeout
        while True:
            for slot_id in range(self.max_slots):
                slot_path = self.slots_dir / f"slot_{slot_id}.lock"
                try:
                    f = open(slot_path, "a+b")
                    try_lock_exclusive(f)
                    with self._mutex:
                        self._local_locks[slot_id] = f
                    return slot_id, f
                except (FileLockUnavailable, OSError):
                    try:
                        f.close()
                    except Exception:
                        pass
                    continue

            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for container capacity slot after {timeout:.1f}s"
                )
            time.sleep(poll_interval)

    def release_slot(self, slot_id: int, file_handle: Any) -> None:
        """Release a previously acquired container capacity slot."""
        with self._mutex:
            self._local_locks.pop(slot_id, None)
        try:
            unlock_exclusive(file_handle)
        except Exception:
            pass
        try:
            file_handle.close()
        except Exception:
            pass

    def active_container_computations(self) -> int:
        """Probe actively executing computations across all worker processes."""
        active = 0
        for slot_id in range(self.max_slots):
            with self._mutex:
                if slot_id in self._local_locks:
                    active += 1
                    continue
            slot_path = self.slots_dir / f"slot_{slot_id}.lock"
            if not slot_path.exists():
                continue
            try:
                f = open(slot_path, "a+b")
                try:
                    try_lock_exclusive(f)
                    unlock_exclusive(f)
                except (FileLockUnavailable, OSError):
                    active += 1
                finally:
                    f.close()
            except OSError:
                pass
        return active


class UploadAdmissionCoordinator:
    """Kernel-advisory-locked slot manager for aggregate concurrent-upload admission control.

    Guarantees:
    - Kernel-enforced upper bound on concurrent uploads across all worker processes using advisory file locks.
    - Automatic release on worker crash/restart (OS kernel closes file handles and releases locks).
    - Immediate rejection without blocking when aggregate limit is reached (HTTP 429).
    """

    def __init__(
        self,
        slots_dir: Optional[Path] = None,
        max_slots: Optional[int] = None,
    ) -> None:
        if max_slots is None:
            max_slots = int(
                os.environ.get(
                    "MAX_CONCURRENT_UPLOADS",
                    str(settings.MAX_CONCURRENT_UPLOADS),
                )
            )
        self.max_slots = max(1, max_slots)

        if slots_dir is None:
            base_dir = (
                Path(settings.CAPACITY_SLOTS_DIR)
                if settings.CAPACITY_SLOTS_DIR
                else (Path(settings.PRIVATE_STORAGE_PATH) / ".capacity_slots")
            )
            self.slots_dir = (base_dir / "uploads").resolve()
        else:
            self.slots_dir = (Path(slots_dir) / "uploads").resolve()

        self.slots_dir.mkdir(parents=True, exist_ok=True)
        self._local_locks: dict[int, Any] = {}
        self._mutex = threading.Lock()

    def acquire_upload_slot(self) -> Optional[Tuple[int, Any]]:
        """Try to acquire an available upload slot immediately without blocking.

        Returns (slot_id, file_handle) if acquired, or None if all slots are occupied.
        """
        for slot_id in range(self.max_slots):
            slot_path = self.slots_dir / f"upload_slot_{slot_id}.lock"
            try:
                f = open(slot_path, "a+b")
                try_lock_exclusive(f)
                with self._mutex:
                    self._local_locks[slot_id] = f
                return slot_id, f
            except (FileLockUnavailable, OSError):
                try:
                    f.close()
                except Exception:
                    pass
                continue
        return None

    def release_upload_slot(self, slot_id: int, file_handle: Any) -> None:
        """Release a previously acquired upload slot."""
        with self._mutex:
            self._local_locks.pop(slot_id, None)
        try:
            unlock_exclusive(file_handle)
        except Exception:
            pass
        try:
            file_handle.close()
        except Exception:
            pass

    def active_upload_count(self) -> int:
        """Probe actively executing uploads across all worker processes."""
        active = 0
        for slot_id in range(self.max_slots):
            with self._mutex:
                if slot_id in self._local_locks:
                    active += 1
                    continue
            slot_path = self.slots_dir / f"upload_slot_{slot_id}.lock"
            if not slot_path.exists():
                continue
            try:
                f = open(slot_path, "a+b")
                try:
                    try_lock_exclusive(f)
                    unlock_exclusive(f)
                except (FileLockUnavailable, OSError):
                    active += 1
                finally:
                    f.close()
            except OSError:
                pass
        return active


class ImageBackendService:
    """Coordinates image assessment, case ownership authorization, and evidence preservation."""

    _shared_limiter: Optional[ProcessCapacityLimiter] = None
    _shared_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _shared_coordinator: Optional[ContainerCapacityCoordinator] = None
    _custom_neural_pipeline: Optional[IntegratedForensicPipeline] = None
    _custom_engine_a: Optional[SecureInferenceEngine] = None
    _custom_service_b: Optional[ModelBFacialInferenceService] = None
    _cached_engine_a: Optional[SecureInferenceEngine] = None
    _cached_service_b: Optional[ModelBFacialInferenceService] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        max_concurrent_analyses: Optional[int] = None,
        timeout_seconds: float = 30.0,
        isolated: bool = False,
        slots_dir: Optional[Path] = None,
        max_container_slots: Optional[int] = None,
    ) -> None:
        if max_concurrent_analyses is None:
            max_concurrent_analyses = int(
                os.environ.get(
                    "IMAGE_CONCURRENCY_SEMAPHORE",
                    str(settings.WORKER_MAX_CONCURRENT_ANALYSES),
                )
            )
        self.max_concurrent_analyses = max_concurrent_analyses
        self.timeout_seconds = timeout_seconds

        if isolated:
            self._limiter = ProcessCapacityLimiter(max_concurrent_analyses)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrent_analyses,
                thread_name_prefix="safeher-vision-worker-iso",
            )
            self._coordinator = ContainerCapacityCoordinator(
                slots_dir=slots_dir,
                max_slots=max_container_slots,
            )
        else:
            with ImageBackendService._init_lock:
                if ImageBackendService._shared_limiter is None:
                    ImageBackendService._shared_limiter = ProcessCapacityLimiter(max_concurrent_analyses)
                if ImageBackendService._shared_executor is None:
                    ImageBackendService._shared_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_concurrent_analyses,
                        thread_name_prefix="safeher-vision-worker",
                    )
                if ImageBackendService._shared_coordinator is None:
                    ImageBackendService._shared_coordinator = ContainerCapacityCoordinator(
                        slots_dir=slots_dir,
                        max_slots=max_container_slots,
                    )
                self._limiter = ImageBackendService._shared_limiter
                self._executor = ImageBackendService._shared_executor
                self._coordinator = ImageBackendService._shared_coordinator

    @classmethod
    def set_neural_pipeline(cls, pipeline: Optional[IntegratedForensicPipeline]) -> None:
        """Inject or override the integrated neural pipeline (primarily for testing)."""
        with cls._init_lock:
            cls._custom_neural_pipeline = pipeline

    @classmethod
    def set_custom_engines(
        cls,
        engine_a: Optional[SecureInferenceEngine] = None,
        service_b: Optional[ModelBFacialInferenceService] = None,
    ) -> None:
        """Inject custom neural inference engines for testing."""
        with cls._init_lock:
            cls._custom_engine_a = engine_a
            cls._custom_service_b = service_b

    @classmethod
    def reset_shared_state(cls) -> None:
        """Helper to reset process-level singletons (used in testing)."""
        with cls._init_lock:
            if cls._shared_executor:
                cls._shared_executor.shutdown(wait=False)
            cls._shared_executor = None
            cls._shared_limiter = None
            cls._shared_coordinator = None
            cls._custom_neural_pipeline = None
            cls._custom_engine_a = None
            cls._custom_service_b = None
            cls._cached_engine_a = None
            cls._cached_service_b = None

    def _validate_and_load_neural_engines(
        self,
    ) -> Tuple[
        Optional[SecureInferenceEngine],
        Optional[ModelBFacialInferenceService],
        str,
        List[str],
        Dict[str, Optional[str]],
    ]:
        with ImageBackendService._init_lock:
            if ImageBackendService._custom_engine_a is not None or ImageBackendService._custom_service_b is not None:
                return (
                    ImageBackendService._custom_engine_a,
                    ImageBackendService._custom_service_b,
                    "active",
                    [],
                    {
                        "model_a": settings.AI_NEURAL_EXPECTED_VERSION_A,
                        "model_b": settings.AI_NEURAL_EXPECTED_VERSION_B,
                    },
                )

        warnings: List[str] = []
        versions: Dict[str, Optional[str]] = {"model_a": None, "model_b": None}
        overall_status = "active"

        # 1. Model A Artifact Verification
        model_a_path_str = settings.AI_NEURAL_MODEL_A_PATH
        engine_a: Optional[SecureInferenceEngine] = None
        if not model_a_path_str:
            warnings.append("AI_NEURAL_MODEL_A_PATH is not configured.")
            overall_status = "model_unavailable"
        else:
            path_a = Path(model_a_path_str).resolve()
            # Allowed directories check (CWE-22 defense)
            is_allowed = False
            for allowed_dir_str in settings.AI_NEURAL_ALLOWED_ARTIFACT_DIRS:
                allowed_dir = Path(allowed_dir_str).resolve()
                try:
                    path_a.relative_to(allowed_dir)
                    is_allowed = True
                    break
                except ValueError:
                    continue

            if not is_allowed:
                warnings.append(
                    f"Model A path '{path_a}' is outside allowed artifact directories (CWE-22 defense)."
                )
                overall_status = "invalid_artifact"
            elif not path_a.is_file():
                warnings.append(f"Model A artifact file not found: '{path_a}'.")
                overall_status = "model_unavailable"
            else:
                # Checksum verification if configured
                if settings.AI_NEURAL_MODEL_A_SHA256:
                    try:
                        computed_sha, _ = compute_streaming_sha256(path_a)
                        if computed_sha.lower() != settings.AI_NEURAL_MODEL_A_SHA256.lower():
                            warnings.append(
                                f"Model A SHA-256 digest mismatch! Expected: {settings.AI_NEURAL_MODEL_A_SHA256}, computed: {computed_sha}."
                            )
                            overall_status = "checksum_mismatch"
                    except Exception as hash_exc:
                        warnings.append(f"Failed to verify Model A hash: {str(hash_exc)}")
                        overall_status = "checksum_mismatch"

                if overall_status == "active":
                    with ImageBackendService._init_lock:
                        if ImageBackendService._cached_engine_a is None:
                            try:
                                metadata_path = path_a.with_suffix(".json")
                                if not metadata_path.exists():
                                    metadata_path = path_a.parent / f"{path_a.stem}_metadata.json"
                                engine_cfg = SecureEngineConfig(
                                    timeout_seconds=settings.AI_NEURAL_INFERENCE_TIMEOUT_SECONDS,
                                    device=settings.AI_DEVICE,
                                    max_batch_size=settings.AI_NEURAL_MAX_BATCH_SIZE,
                                )
                                ImageBackendService._cached_engine_a = SecureInferenceEngine(
                                    artifact_path=path_a,
                                    metadata_path=metadata_path if metadata_path.exists() else path_a,
                                    config=engine_cfg,
                                    lazy_load=True,
                                )
                            except Exception as engine_exc:
                                logger.warning("Failed to initialize Model A engine: %s", engine_exc)
                                warnings.append(f"Failed to initialize Model A engine: {str(engine_exc)}")
                                overall_status = "invalid_artifact"
                        engine_a = ImageBackendService._cached_engine_a
                        if engine_a is not None:
                            versions["model_a"] = settings.AI_NEURAL_EXPECTED_VERSION_A

        # 2. Model B Artifact Verification
        model_b_path_str = settings.AI_NEURAL_MODEL_B_PATH
        service_b: Optional[ModelBFacialInferenceService] = None
        if model_b_path_str:
            path_b = Path(model_b_path_str).resolve()
            is_allowed_b = False
            for allowed_dir_str in settings.AI_NEURAL_ALLOWED_ARTIFACT_DIRS:
                allowed_dir = Path(allowed_dir_str).resolve()
                try:
                    path_b.relative_to(allowed_dir)
                    is_allowed_b = True
                    break
                except ValueError:
                    continue

            if not is_allowed_b:
                warnings.append(
                    f"Model B path '{path_b}' is outside allowed artifact directories."
                )
                if overall_status == "active":
                    overall_status = "invalid_artifact"
            elif not path_b.is_file():
                warnings.append(f"Model B artifact file not found: '{path_b}'.")
                if overall_status == "active":
                    overall_status = "model_unavailable"
            else:
                if settings.AI_NEURAL_MODEL_B_SHA256:
                    try:
                        computed_sha_b, _ = compute_streaming_sha256(path_b)
                        if computed_sha_b.lower() != settings.AI_NEURAL_MODEL_B_SHA256.lower():
                            warnings.append(
                                f"Model B SHA-256 digest mismatch! Expected: {settings.AI_NEURAL_MODEL_B_SHA256}, computed: {computed_sha_b}."
                            )
                            if overall_status == "active":
                                overall_status = "checksum_mismatch"
                    except Exception as hash_exc:
                        warnings.append(f"Failed to verify Model B hash: {str(hash_exc)}")
                        if overall_status == "active":
                            overall_status = "checksum_mismatch"

                if overall_status == "active":
                    with ImageBackendService._init_lock:
                        if ImageBackendService._cached_service_b is None:
                            try:
                                ImageBackendService._cached_service_b = ModelBFacialInferenceService(
                                    checkpoint_path=path_b,
                                    expected_sha256=settings.AI_NEURAL_MODEL_B_SHA256,
                                    enforce_trained=True,
                                )
                            except Exception as b_exc:
                                logger.warning("Failed to initialize Model B service: %s", b_exc)
                                warnings.append(f"Failed to initialize Model B service: {str(b_exc)}")
                        service_b = ImageBackendService._cached_service_b
                        if service_b is not None:
                            versions["model_b"] = settings.AI_NEURAL_EXPECTED_VERSION_B

        return engine_a, service_b, overall_status, warnings, versions

    def _execute_assessment(self, raw_bytes: bytes) -> IntegratedForensicAssessment:
        """Execute baseline or integrated neural assessment with privacy and safety guards."""
        # 1. Custom pipeline override (primarily for unit/integration testing)
        with ImageBackendService._init_lock:
            custom_pipeline = ImageBackendService._custom_neural_pipeline

        if custom_pipeline is not None:
            pipeline_res = custom_pipeline.analyze(data=raw_bytes)
            cal_status = "NOT_CALIBRATED"
            if pipeline_res.ai_generated_result and pipeline_res.ai_generated_result.calibrated_probability is not None:
                cal_status = "CALIBRATED_TEMPERATURE_SCALING"
            neural_meta = NeuralInferenceMetadata(
                status="active" if custom_pipeline.enable_neural else "disabled",
                model_a_version=settings.AI_NEURAL_EXPECTED_VERSION_A if custom_pipeline.enable_neural else None,
                model_b_version=settings.AI_NEURAL_EXPECTED_VERSION_B if custom_pipeline.enable_neural else None,
                supported_tasks=[
                    "AI_GENERATED_IMAGE_DETECTION",
                    "FACIAL_MANIPULATION_DETECTION",
                ],
                uncertainty_state=getattr(pipeline_res.uncertainty_state, "value", str(pipeline_res.uncertainty_state)),
                calibration_status=cal_status,
                warnings=[],
            )
            res_dict = pipeline_res.model_dump()
            res_dict["neural_metadata"] = neural_meta
            return IntegratedForensicAssessment.model_validate(res_dict)

        # 2. Neural feature disabled: baseline heuristic forensics
        if not settings.AI_ENABLE_NEURAL_DETECTOR:
            base_resp = ai_image_service.assess_image(raw_bytes, None)
            resp_dict = base_resp.model_dump()
            resp_dict["is_neural_enabled"] = False
            resp_dict["neural_metadata"] = NeuralInferenceMetadata(
                status="disabled",
                model_a_version=None,
                model_b_version=None,
                supported_tasks=[
                    "AI_GENERATED_IMAGE_DETECTION",
                    "FACIAL_MANIPULATION_DETECTION",
                ],
                uncertainty_state=getattr(base_resp.uncertainty_state, "value", str(base_resp.uncertainty_state)),
                calibration_status="NOT_APPLICABLE",
                warnings=["Neural detection feature flag (AI_ENABLE_NEURAL_DETECTOR) is disabled."],
            )
            return IntegratedForensicAssessment.model_validate(resp_dict)

        # 3. Pre-validate image bytes to reject corrupted or invalid formats immediately
        preprocess_image(data=raw_bytes)

        # 4. Neural feature enabled: validate and load models
        engine_a, service_b, load_status, load_warnings, versions = self._validate_and_load_neural_engines()

        if load_status != "active":
            logger.warning(
                "Neural detector unavailable (%s): %s. Executing baseline fallback.",
                load_status,
                "; ".join(load_warnings),
            )
            base_resp = ai_image_service.assess_image(raw_bytes, None)
            resp_dict = base_resp.model_dump()
            resp_dict["is_neural_enabled"] = False
            resp_dict["neural_metadata"] = NeuralInferenceMetadata(
                status=load_status,
                model_a_version=versions.get("model_a"),
                model_b_version=versions.get("model_b"),
                supported_tasks=[
                    "AI_GENERATED_IMAGE_DETECTION",
                    "FACIAL_MANIPULATION_DETECTION",
                ],
                uncertainty_state="INCONCLUSIVE" if load_status in ("invalid_artifact", "checksum_mismatch") else "ELEVATED_UNCERTAINTY",
                calibration_status="UNAVAILABLE",
                warnings=load_warnings,
            )
            return IntegratedForensicAssessment.model_validate(resp_dict)

        # 5. Execute neural integrated pipeline
        try:
            pipeline = IntegratedForensicPipeline(
                enable_neural=True,
                model_a_engine=engine_a,
                model_b_service=service_b,
            )
            result = pipeline.analyze(data=raw_bytes)

            cal_status = "NOT_CALIBRATED"
            if result.ai_generated_result and result.ai_generated_result.calibrated_probability is not None:
                cal_status = "CALIBRATED_TEMPERATURE_SCALING"

            neural_status = "active"
            if result.ai_generated_result and result.ai_generated_result.decision_state == "TIMEOUT":
                neural_status = "timeout"

            neural_meta = NeuralInferenceMetadata(
                status=neural_status,
                model_a_version=versions.get("model_a"),
                model_b_version=versions.get("model_b"),
                supported_tasks=[
                    "AI_GENERATED_IMAGE_DETECTION",
                    "FACIAL_MANIPULATION_DETECTION",
                ],
                uncertainty_state=getattr(result.uncertainty_state, "value", str(result.uncertainty_state)),
                calibration_status=cal_status,
                warnings=load_warnings,
            )
            res_dict = result.model_dump()
            res_dict["neural_metadata"] = neural_meta
            return IntegratedForensicAssessment.model_validate(res_dict)
        except (InferenceTimeoutError, TimeoutError) as timeout_exc:
            logger.error("Neural inference execution timed out: %s", timeout_exc)
            base_resp = ai_image_service.assess_image(raw_bytes, None)
            resp_dict = base_resp.model_dump()
            resp_dict["is_neural_enabled"] = True
            resp_dict["assessment_label"] = "inconclusive"
            resp_dict["uncertainty_state"] = "INCONCLUSIVE"
            resp_dict["neural_metadata"] = NeuralInferenceMetadata(
                status="timeout",
                model_a_version=versions.get("model_a"),
                model_b_version=versions.get("model_b"),
                supported_tasks=[
                    "AI_GENERATED_IMAGE_DETECTION",
                    "FACIAL_MANIPULATION_DETECTION",
                ],
                uncertainty_state="INCONCLUSIVE",
                calibration_status="UNAVAILABLE",
                warnings=["Neural inference exceeded allotted execution deadline."],
            )
            return IntegratedForensicAssessment.model_validate(resp_dict)
        except Exception as exec_err:
            logger.error("Unexpected error in neural pipeline: %s", exec_err)
            base_resp = ai_image_service.assess_image(raw_bytes, None)
            resp_dict = base_resp.model_dump()
            resp_dict["is_neural_enabled"] = False
            resp_dict["neural_metadata"] = NeuralInferenceMetadata(
                status="error",
                model_a_version=versions.get("model_a"),
                model_b_version=versions.get("model_b"),
                supported_tasks=[
                    "AI_GENERATED_IMAGE_DETECTION",
                    "FACIAL_MANIPULATION_DETECTION",
                ],
                uncertainty_state="INCONCLUSIVE",
                calibration_status="UNAVAILABLE",
                warnings=[f"Neural execution error: {str(exec_err)}"],
            )
            return IntegratedForensicAssessment.model_validate(resp_dict)

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with ImageBackendService._init_lock:
            if self._executor is None or getattr(self._executor, "_shutdown", False):
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_concurrent_analyses,
                    thread_name_prefix="safeher-vision-worker",
                )
                ImageBackendService._shared_executor = self._executor
            return self._executor

    def _get_limiter(self) -> ProcessCapacityLimiter:
        with ImageBackendService._init_lock:
            if self._limiter is None or ImageBackendService._shared_limiter is None:
                self._limiter = ProcessCapacityLimiter(self.max_concurrent_analyses)
                ImageBackendService._shared_limiter = self._limiter
            return self._limiter

    @property
    def active_computations(self) -> int:
        return self._get_limiter().active_count

    @property
    def available_permits(self) -> int:
        return self._get_limiter().available_permits

    @property
    def active_container_computations(self) -> int:
        return self._coordinator.active_container_computations()

    @property
    def available_container_slots(self) -> int:
        return max(0, self._coordinator.max_slots - self._coordinator.active_container_computations())

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    async def assess_image_upload(
        self,
        file: UploadFile,
        current_user: User,
        case_id: Optional[UUID] = None,
        db: Optional[Session] = None,
    ) -> IntegratedForensicAssessment:
        """Securely validate, process, and triage an uploaded image.

        Args:
            file: Incoming multipart UploadFile stream.
            current_user: Authenticated user initiating the assessment.
            case_id: Optional case ID to link evidence and audit logging.
            db: Database session for case verification and evidence persistence.

        Returns:
            Structured IntegratedForensicAssessment.

        Raises:
            HTTPException: 400 for payload/validation/size errors.
            HTTPException: 403 for unauthorized case access.
            HTTPException: 404 for non-existent case.
            HTTPException: 500 for internal runtime failures.
        """
        # 1. Authorize case ownership if case_id provided
        linked_case: Optional[Case] = None
        if case_id is not None:
            if db is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session required for case verification.",
                )

            linked_case = db.scalar(select(Case).where(Case.id == case_id))
            if linked_case is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Case not found.",
                )

            if linked_case.owner_id != current_user.id:
                logger.warning(
                    "Unauthorized case link attempt for image: user_id=%s, case_id=%s",
                    current_user.id,
                    case_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this case.",
                )

        # 2. Early filename sanitization and extension validation
        raw_filename = (file.filename or "uploaded_image.jpg").replace("\x00", "")
        # Path.name on POSIX does not treat backslashes as separators, so normalize
        # both styles before validating or persisting user-supplied filename metadata.
        safe_filename = raw_filename.replace("\\", "/").rsplit("/", 1)[-1][:255]
        if not safe_filename:
            safe_filename = "uploaded_image.jpg"
        suffix = Path(safe_filename).suffix.lower()
        if suffix and suffix not in ALLOWED_IMAGE_EXTENSIONS:
            logger.warning(
                "Unsupported file extension rejected: user_id=%s, filename=%s, ext=%s",
                current_user.id,
                safe_filename,
                suffix,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported or invalid image file format: Unsupported file extension '{suffix}'. Allowed extensions: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
            )

        # 3. Chunked stream read with strict size enforcement
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        chunk_size = 1024 * 1024  # 1 MB chunk
        chunks = []
        total_bytes = 0

        try:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    logger.warning(
                        "Oversized image upload rejected: user_id=%s, total_bytes=%d, max_allowed=%d",
                        current_user.id,
                        total_bytes,
                        max_bytes,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image upload exceeds the maximum allowable size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                    )
                chunks.append(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to read image upload stream: %s", str(exc))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to read uploaded image stream.",
            ) from exc
        finally:
            await file.close()

        raw_bytes = b"".join(chunks)
        if not raw_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file cannot be empty.",
            )

        # 4. Offload CPU-bound computer vision assessment to bounded thread pool with capacity-retaining permit
        start_time = time.perf_counter()

        limiter = self._get_limiter()
        executor = self._get_executor()

        # Acquire capacity permit before submitting to worker thread
        await limiter.acquire()

        release_lock = threading.Lock()
        permit_released = False
        slot_holder: list[Tuple[int, Any]] = []

        def _release_resources_once() -> None:
            nonlocal permit_released
            with release_lock:
                if not permit_released:
                    permit_released = True
                    if slot_holder:
                        slot_id, handle = slot_holder[0]
                        self._coordinator.release_slot(slot_id, handle)
                    limiter.release()

        def _compute_in_worker() -> IntegratedForensicAssessment:
            try:
                slot_id, handle = self._coordinator.acquire_slot(timeout=self.timeout_seconds)
                slot_holder.append((slot_id, handle))
                return self._execute_assessment(raw_bytes)
            finally:
                _release_resources_once()

        try:
            concurrent_future = executor.submit(_compute_in_worker)
        except Exception:
            _release_resources_once()
            raise

        async_future = asyncio.wrap_future(concurrent_future)

        def _consume_detached_result(done_future: asyncio.Future) -> None:
            # If a client disconnects or the HTTP request times out, the worker
            # still runs to completion so its permit/slot can be released safely.
            # Retrieving the exception prevents an unobserved-future warning.
            if not done_future.cancelled():
                try:
                    done_future.exception()
                except Exception:
                    pass

        async_future.add_done_callback(_consume_detached_result)
        try:
            response: IntegratedForensicAssessment = await asyncio.wait_for(
                asyncio.shield(async_future),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "Image assessment timed out after %.1fs for user_id=%s. Capacity permit retained until worker genuinely finishes.",
                self.timeout_seconds,
                current_user.id,
            )
            model_monitoring_engine.record_assessment_telemetry(
                response=None,
                is_failure=True,
                failure_category="timeout",
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Image assessment timed out. Please try again with an optimized image.",
            ) from exc
        except ValueError as exc:
            # Preprocessing rejection (magic bytes mismatch, corrupt stream, dimension violation)
            logger.warning(
                "Image assessment input rejected: user_id=%s, reason=%s",
                current_user.id,
                str(exc),
            )
            model_monitoring_engine.record_assessment_telemetry(
                response=None,
                is_failure=True,
                failure_category="validation_error",
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image: {str(exc)}",
            ) from exc
        except Exception as exc:
            # Catch unexpected internal runtime errors; NEVER log raw image content
            logger.exception("Unexpected error during image assessment for user_id=%s", current_user.id)
            model_monitoring_engine.record_assessment_telemetry(
                response=None,
                is_failure=True,
                failure_category="internal_error",
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during image assessment. Please try again.",
            ) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Secure evidence preservation if linked to an existing case
        if case_id is not None and db is not None:
            # Sanitize original filename (prevent path traversal)
            safe_original_filename = safe_filename

            # Generate random server-side storage key
            storage_key = f"{uuid4()}.bin"

            # Register database-backed ACTIVE_UPLOAD lease to protect against concurrent orphan cleanup
            upload_lease = EvidenceCleanupLease(
                storage_key=storage_key,
                status="ACTIVE_UPLOAD",
                action="upload",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
                notes=f"Active evidence upload in progress for case {case_id}",
            )
            try:
                db.add(upload_lease)
                db.commit()
            except Exception as lease_exc:
                db.rollback()
                logger.error("Database failure while registering upload lease: %s", str(lease_exc))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database error while persisting evidence record.",
                ) from lease_exc

            # Enforce path traversal prevention and resolve storage directory
            storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
            storage_dir.mkdir(parents=True, exist_ok=True)

            dest_path = (storage_dir / storage_key).resolve()
            try:
                dest_path.relative_to(storage_dir)
            except ValueError as path_exc:
                upload_lease.status = "ABORTED"
                upload_lease.notes = "Generated evidence path escaped the private storage directory"
                db.commit()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid storage destination.",
                ) from path_exc

            # Write pristine evidence bytes via temporary .part file and atomic rename
            part_path = dest_path.with_suffix(".part")
            try:
                part_path.write_bytes(raw_bytes)
                part_path.replace(dest_path)  # Atomic rename
            except Exception as exc:
                if part_path.exists():
                    part_path.unlink(missing_ok=True)
                upload_lease.status = "ABORTED"
                upload_lease.notes = f"Filesystem write failed: {str(exc)[:400]}"
                db.commit()
                logger.error("Failed to write evidence to private storage: %s", str(exc))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to persist evidence securely.",
                ) from exc

            sha256_hash = (
                response.metadata_summary.sha256
                if response.metadata_summary
                else hashlib.sha256(raw_bytes).hexdigest()
            )
            mime_type = (
                f"image/{response.metadata_summary.format.lower()}"
                if response.metadata_summary
                else "image/jpeg"
            )

            try:
                # Pre-commit verification: ensure orphan cleanup hasn't claimed or quarantined this file
                cleanup_conflict = db.scalar(
                    select(EvidenceCleanupLease).where(
                        EvidenceCleanupLease.storage_key == storage_key,
                        EvidenceCleanupLease.status.in_(["QUARANTINED", "PURGED"]),
                    )
                )
                if cleanup_conflict is not None:
                    raise RuntimeError("Cleanup lease conflict intercepted before evidence commit.")

                evidence_record = Evidence(
                    case_id=case_id,
                    owner_id=current_user.id,
                    original_filename=safe_original_filename,
                    storage_key=storage_key,
                    mime_type=mime_type,
                    size_bytes=len(raw_bytes),
                    sha256=sha256_hash,
                )
                db.add(evidence_record)
                db.flush()

                # Mark active upload lease as completed
                upload_lease.status = "COMPLETED"
                upload_lease.resolved_at = datetime.now(timezone.utc)
                upload_lease.notes = "Evidence successfully persisted and linked"

                # Link evidence ID into response
                response.image_id = evidence_record.id

                audit = AuditLog(
                    user_id=current_user.id,
                    action="AI_IMAGE_ASSESSMENT",
                    entity_type="case",
                    entity_id=str(case_id),
                )
                db.add(audit)
                db.commit()
            except Exception as db_exc:
                db.rollback()
                try:
                    upload_lease.status = "ABORTED"
                    upload_lease.resolved_at = datetime.now(timezone.utc)
                    db.commit()
                except Exception:
                    pass
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                if part_path.exists():
                    try:
                        part_path.unlink()
                    except Exception:
                        pass
                logger.error("Database failure while persisting evidence record: %s", str(db_exc))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database error while persisting evidence record.",
                ) from db_exc

        # 5. Privacy-safe telemetry logging: log ONLY non-sensitive metrics
        logger.info(
            "Image assessment completed: assessment_id=%s, user_id=%s, label=%s, raw_score=%.2f, regions_count=%d, elapsed_ms=%.2f",
            response.assessment_id,
            current_user.id,
            response.assessment_label.value,
            response.raw_model_score,
            len(response.manipulation_regions),
            elapsed_ms,
        )

        model_monitoring_engine.record_assessment_telemetry(
            response=response,
            is_failure=False,
            elapsed_ms=elapsed_ms,
            image_format=response.metadata_summary.format if response.metadata_summary else "UNKNOWN",
            dimensions=response.metadata_summary.dimensions if response.metadata_summary else (0, 0),
        )

        return response


backend_image_service = ImageBackendService()
upload_admission_coordinator = UploadAdmissionCoordinator()
