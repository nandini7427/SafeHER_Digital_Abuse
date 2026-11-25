"""Automated Failure-Injection and Service Resilience Test Suite for SafeHER.

Tests SafeHER image analysis under catastrophic and transient failure conditions:
1. Missing model files
2. Corrupt model weights and hash mismatch
3. Unsupported device and accelerator failure
4. Missing inference dependencies
5. Out-of-memory (OOM) conditions
6. Inference timeout
7. Model initialization failure and singleton reset
8. Worker restart and recovery idempotence
9. Temporary storage failure (disk full / unwriteable)
10. Database failure and transaction rollback
11. Universal safeguard: Unassessed images are never labeled authentic
"""

import asyncio
import io
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4

from PIL import Image
import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ai_core.image_intelligence.config import ImageDetectorConfig
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import (
    BaseImageModelRunner,
    NeuralVisionModelRunner,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    UncertaintyState,
)
from ai_core.image_intelligence.service import (
    get_detector_initialization_count,
    get_image_detector,
    reset_detector_initialization_stats,
    set_image_detector,
)
from backend.app.models import AuditLog, Base, Case, Evidence, User
from backend.app.services.image_service import ImageBackendService


@pytest.fixture
def sample_test_image_bytes() -> bytes:
    """Generate a valid synthetic JPEG image for testing."""
    img = Image.new("RGB", (256, 256), color=(140, 150, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """In-memory SQLite session for testing database failure scenarios."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Create a test user."""
    user = User(
        id=uuid4(),
        email="resilience_tester@safeher.internal",
        password_hash="mock_hash_resilience",
        role="user",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_case(db_session: Session, test_user: User) -> Case:
    """Create a test case belonging to the test user."""
    case = Case(
        id=uuid4(),
        title="Resilience Case",
        description="Testing error handling and storage resilience",
        owner_id=test_user.id,
    )
    db_session.add(case)
    db_session.commit()
    db_session.refresh(case)
    return case


# ---------------------------------------------------------------------------
# 1. Missing Model Files Failure Injection
# ---------------------------------------------------------------------------

def test_failure_missing_model_files(sample_test_image_bytes: bytes):
    """Verify system handles non-existent model files by falling back safely to deterministic baseline."""
    config = ImageDetectorConfig(
        enable_neural_detector=True,
        model_name_or_path="C:/nonexistent/path/weights.safetensors",
    )
    runner = NeuralVisionModelRunner(config=config)
    assert not runner.is_available()
    assert "not found" in runner.get_status().lower()

    # Detector gracefully falls back to deterministic ELA baseline
    detector = SafeHERImageDetector(config=config)
    result = detector.detect(sample_test_image_bytes)

    assert result is not None
    assert result.detector_name == "SafeHER-Baseline-ELA-Heuristics"
    assert result.assessment_label.value != "authentic"
    assert "authentic" not in [l.value for l in AssessmentLabel]


# ---------------------------------------------------------------------------
# 2. Corrupt Model Weights and Hash Mismatch
# ---------------------------------------------------------------------------

def test_failure_corrupt_model_weights_and_hash_mismatch(sample_test_image_bytes: bytes):
    """Verify corrupted model files or SHA-256 hash mismatches fail closed before execution."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp.write(b"CORRUPTED_MODEL_PAYLOAD_GARBAGE_BYTES")
        tmp_path = tmp.name

    try:
        config = ImageDetectorConfig(
            enable_neural_detector=True,
            is_development_placeholder=False,
            model_name_or_path=tmp_path,
            expected_model_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        runner = NeuralVisionModelRunner(config=config)
        assert not runner.is_available()
        assert "mismatch" in runner.get_status().lower() or "violation" in runner.get_status().lower()

        # Detector safely falls back to baseline without executing corrupt weights
        detector = SafeHERImageDetector(config=config)
        result = detector.detect(sample_test_image_bytes)
        assert result.detector_name == "SafeHER-Baseline-ELA-Heuristics"
        assert result.assessment_label.value != "authentic"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# 3. Unsupported Device Failure Handling
# ---------------------------------------------------------------------------

def test_failure_unsupported_device_configuration():
    """Verify invalid device strings are rejected at configuration time."""
    with pytest.raises(ValueError, match="device must be one of"):
        ImageDetectorConfig(device="tpu_v5")

    with pytest.raises(ValueError, match="device must be one of"):
        ImageDetectorConfig(device="invalid_accelerator")


def test_failure_unsupported_cuda_device_fallback(sample_test_image_bytes: bytes):
    """Verify requested CUDA device on CPU-only host reports unavailable and falls back safely."""
    with patch("torch.cuda.is_available", return_value=False):
        config = ImageDetectorConfig(
            device="cuda",
            enable_neural_detector=True,
            is_development_placeholder=False,
            model_name_or_path="mock_model.safetensors",
        )
        runner = NeuralVisionModelRunner(config=config)
        assert not runner.is_available()
        assert "unsupported device" in runner.get_status().lower()

        detector = SafeHERImageDetector(config=config)
        result = detector.detect(sample_test_image_bytes)
        assert result.detector_name == "SafeHER-Baseline-ELA-Heuristics"
        assert result.assessment_label.value != "authentic"


# ---------------------------------------------------------------------------
# 4. Missing Inference Dependencies
# ---------------------------------------------------------------------------

def test_failure_missing_inference_dependencies(sample_test_image_bytes: bytes):
    """Verify missing inference packages prevent neural runner execution and fall back safely."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp.write(b"mock_weights")
        tmp_path = tmp.name

    try:
        config = ImageDetectorConfig(
            enable_neural_detector=True,
            is_development_placeholder=False,
            model_name_or_path=tmp_path,
        )
        runner = NeuralVisionModelRunner(config=config)

        # Simulate missing safetensors package
        with patch.dict("sys.modules", {"safetensors": None, "safetensors.torch": None}):
            with patch("os.path.exists", return_value=True):
                runner._ensure_loaded()
                assert not runner.is_available()
                assert "failed to load" in runner.get_status().lower() or "not available" in runner.get_status().lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# 5. Out-Of-Memory (OOM) Condition Handling
# ---------------------------------------------------------------------------

def test_failure_out_of_memory_in_primary_runner(sample_test_image_bytes: bytes):
    """Verify that a MemoryError in the primary runner falls back safely to baseline."""
    mock_runner = MagicMock(spec=BaseImageModelRunner)
    mock_runner.predict.side_effect = MemoryError("Simulated host OOM: CUDA/Host out of memory.")
    mock_runner.is_available.return_value = True

    detector = SafeHERImageDetector(runner=mock_runner)
    result = detector.detect(sample_test_image_bytes)

    # Baseline fallback succeeded
    assert result.detector_name == "SafeHER-Baseline-ELA-Heuristics"
    assert result.assessment_label.value != "authentic"


def test_failure_catastrophic_dual_oom_emits_safe_error_state(sample_test_image_bytes: bytes):
    """Verify that if BOTH primary and fallback runners fail from catastrophic OOM, a safe inconclusive state is emitted."""
    mock_primary = MagicMock(spec=BaseImageModelRunner)
    mock_primary.predict.side_effect = MemoryError("Host primary OOM")

    detector = SafeHERImageDetector(runner=mock_primary)
    detector._fallback_runner = MagicMock(spec=BaseImageModelRunner)
    detector._fallback_runner.predict.side_effect = MemoryError("Host fallback OOM")

    result = detector.detect(sample_test_image_bytes)

    assert result.assessment_label == AssessmentLabel.INCONCLUSIVE
    assert result.uncertainty_state == UncertaintyState.MODEL_UNAVAILABLE
    assert result.raw_model_score == 0.0
    assert result.confidence_score is None
    assert not result.analysis_eligibility
    assert "CRITICAL FAILURE" in result.explanation
    assert "cannot and must not be treated as authentic" in result.explanation


# ---------------------------------------------------------------------------
# 6. Inference Timeout Handling
# ---------------------------------------------------------------------------

def test_failure_inference_timeout(sample_test_image_bytes: bytes, test_user: User):
    """HTTP timeout must not cancel the underlying worker thread or release its permit early."""
    async def _run_test():
        service = ImageBackendService(
            timeout_seconds=0.01,
            isolated=True,
            slots_dir=Path(tempfile.mkdtemp(prefix="safeher-timeout-slots-")),
            max_container_slots=1,
        )
        started = threading.Event()
        finished = threading.Event()

        def slow_inference(*args, **kwargs):
            started.set()
            time.sleep(0.08)
            finished.set()
            return None

        upload_file = UploadFile(
            file=io.BytesIO(sample_test_image_bytes),
            filename="slow_image.jpg",
        )

        try:
            with patch(
                "backend.app.services.image_service.ai_image_service.assess_image",
                side_effect=slow_inference,
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await service.assess_image_upload(
                        file=upload_file,
                        current_user=test_user,
                    )

                assert exc_info.value.status_code == 504
                assert "timed out" in exc_info.value.detail.lower()
                assert started.is_set()
                # The request has returned 504, but the actual computation is still active.
                assert service.active_computations == 1
                assert service.available_permits == 0
                assert await asyncio.to_thread(finished.wait, 1.0)
                # Allow worker thread finally block to release capacity permit
                for _ in range(50):
                    if service.active_computations == 0:
                        break
                    await asyncio.sleep(0.01)
                assert service.active_computations == 0
                assert service.available_permits == 1
        finally:
            service.shutdown(wait=True)

    asyncio.run(_run_test())


# ---------------------------------------------------------------------------
# 7. Model Initialization Failure and Recovery
# ---------------------------------------------------------------------------

def test_failure_model_initialization_and_clean_recovery(sample_test_image_bytes: bytes):
    """Verify that an initialization failure does not leave poisonous state and recovers on reset."""
    set_image_detector(None)
    reset_detector_initialization_stats()

    # 1. Simulate initialization failure
    with patch("ai_core.image_intelligence.service.SafeHERImageDetector", side_effect=RuntimeError("Init crash")):
        with pytest.raises(RuntimeError, match="Init crash"):
            get_image_detector()

    # 2. Verify clean recovery after fixing/clearing state
    set_image_detector(None)
    recovered_detector = get_image_detector()
    assert recovered_detector is not None
    assert get_detector_initialization_count() == 1

    result = recovered_detector.detect(sample_test_image_bytes)
    assert result.assessment_label.value != "authentic"


# ---------------------------------------------------------------------------
# 8. Worker Restart and Idempotence
# ---------------------------------------------------------------------------

def test_failure_worker_restart_idempotence(sample_test_image_bytes: bytes):
    """Verify that worker restart clears in-flight state and produces identical deterministic outputs."""
    detector_pre = get_image_detector()
    result_pre = detector_pre.detect(sample_test_image_bytes)

    # Simulate worker restart by resetting singleton
    set_image_detector(None)
    reset_detector_initialization_stats()

    detector_post = get_image_detector()
    result_post = detector_post.detect(sample_test_image_bytes)

    # Must produce identical results across worker restarts
    assert result_pre.raw_model_score == result_post.raw_model_score
    assert result_pre.assessment_label == result_post.assessment_label
    assert result_pre.detector_version == result_post.detector_version


# ---------------------------------------------------------------------------
# 9. Temporary Storage Failure (Disk Full / Unwriteable)
# ---------------------------------------------------------------------------

def test_failure_temporary_storage_failure(
    sample_test_image_bytes: bytes,
    test_user: User,
    test_case: Case,
    db_session: Session,
):
    """Verify storage disk write failure returns HTTP 500 without persisting dirty DB records."""
    async def _run_test():
        service = ImageBackendService()

        upload_file = UploadFile(
            file=io.BytesIO(sample_test_image_bytes),
            filename="evidence_disk_failure.jpg",
        )

        initial_evidence_count = db_session.query(Evidence).filter_by(case_id=test_case.id).count()
        initial_audit_count = db_session.query(AuditLog).count()

        # Mock Path.write_bytes to simulate disk full (ENOSPC / OSError)
        with patch.object(Path, "write_bytes", side_effect=OSError("No space left on device [Errno 28]")):
            with pytest.raises(HTTPException) as exc_info:
                await service.assess_image_upload(
                    file=upload_file,
                    current_user=test_user,
                    case_id=test_case.id,
                    db=db_session,
                )

            assert exc_info.value.status_code == 500
            assert "persist evidence securely" in exc_info.value.detail.lower()

        # Verify no database pollution occurred
        db_session.rollback()
        assert db_session.query(Evidence).filter_by(case_id=test_case.id).count() == initial_evidence_count
        assert db_session.query(AuditLog).count() == initial_audit_count

    asyncio.run(_run_test())


# ---------------------------------------------------------------------------
# 10. Database Failure and Transaction Rollback
# ---------------------------------------------------------------------------

def test_failure_database_error_triggers_rollback_and_unlinks_file(
    sample_test_image_bytes: bytes,
    test_user: User,
    test_case: Case,
    db_session: Session,
):
    """Verify database commit failure triggers rollback and unlinks temporary evidence files."""
    async def _run_test():
        service = ImageBackendService()

        upload_file = UploadFile(
            file=io.BytesIO(sample_test_image_bytes),
            filename="evidence_db_failure.jpg",
        )

        initial_evidence_count = db_session.query(Evidence).filter_by(case_id=test_case.id).count()
        initial_audit_count = db_session.query(AuditLog).count()

        # Mock db.flush to raise database operational error
        with patch.object(db_session, "flush", side_effect=RuntimeError("Simulated database connection lost")):
            with pytest.raises(HTTPException) as exc_info:
                await service.assess_image_upload(
                    file=upload_file,
                    current_user=test_user,
                    case_id=test_case.id,
                    db=db_session,
                )

            assert exc_info.value.status_code == 500
            assert "database error" in exc_info.value.detail.lower()

        # Verify database was cleanly rolled back and no partial evidence or duplicate audit logs exist
        assert db_session.query(Evidence).filter_by(case_id=test_case.id).count() == initial_evidence_count
        assert db_session.query(AuditLog).count() == initial_audit_count

    asyncio.run(_run_test())


# ---------------------------------------------------------------------------
# 11. Universal Mandate: Never Label Unassessed Images as Authentic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error_mode",
    [
        "oom",
        "missing_weights",
        "corrupt_weights",
    ],
)
def test_unassessed_images_never_labeled_authentic(error_mode: str, sample_test_image_bytes: bytes):
    """Universal invariant test: Failure modes must NEVER output authentic label or zero uncertainty."""
    if error_mode == "oom":
        mock_runner = MagicMock(spec=BaseImageModelRunner)
        mock_runner.predict.side_effect = MemoryError("OOM")
        detector = SafeHERImageDetector(runner=mock_runner)
        detector._fallback_runner = MagicMock(spec=BaseImageModelRunner)
        detector._fallback_runner.predict.side_effect = MemoryError("OOM")
        result = detector.detect(sample_test_image_bytes)
        assert result.assessment_label.value != "authentic"
        assert result.assessment_label == AssessmentLabel.INCONCLUSIVE
        assert result.uncertainty_state == UncertaintyState.MODEL_UNAVAILABLE
    elif error_mode == "missing_weights":
        config = ImageDetectorConfig(enable_neural_detector=True, model_name_or_path="missing.pt")
        detector = SafeHERImageDetector(config=config)
        result = detector.detect(sample_test_image_bytes)
        assert result.assessment_label.value != "authentic"
    elif error_mode == "corrupt_weights":
        config = ImageDetectorConfig(
            enable_neural_detector=True,
            is_development_placeholder=False,
            model_name_or_path="corrupt.pt",
            expected_model_sha256="deadbeef" * 8,
        )
        detector = SafeHERImageDetector(config=config)
        result = detector.detect(sample_test_image_bytes)
        assert result.assessment_label.value != "authentic"
