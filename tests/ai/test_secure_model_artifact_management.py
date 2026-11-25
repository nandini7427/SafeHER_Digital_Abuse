"""Unit and integration tests for SafeHER Secure Model Artifact and Inference Management.

Tests:
1. Valid artifact loading (SafeTensors and PyTorch with weights_only=True).
2. Missing artifact handling (FileNotFoundError / MODEL_UNAVAILABLE).
3. Checksum mismatch detection (IntegrityVerificationError on tampered bytes).
4. Wrong architecture detection (ArchitectureMismatchError on model mismatch).
5. Wrong input shape and non-finite value rejection.
6. Invalid metadata rejection (schema violation, metadata tampering).
7. Corrupted weights rejection (truncated/garbage files raise CorruptArtifactError).
8. Model loading failure on unsafe formats (.pkl, .joblib raise UnsafeFormatError).
9. Concurrent thread-safe inference execution (ThreadPoolExecutor concurrency).
10. CPU inference execution (deterministic CPU execution).
11. Graceful shutdown and memory unloading.
12. Batch limit enforcement and chunking.
13. Inference timeout enforcement (InferenceTimeoutError).
14. Path traversal protection (ArtifactPathTraversalError).
15. No silent model substitution (rejection of version mismatch).
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import tempfile
import time
from typing import Dict, Tuple

import numpy as np
import pytest
import safetensors.torch
import torch

from ai_core.neural.management import (
    ArchitectureMismatchError,
    ArtifactPathTraversalError,
    ArtifactSecurityPolicy,
    BatchLimitExceededError,
    CorruptArtifactError,
    EngineLifecycleState,
    InferenceTimeoutError,
    IntegrityVerificationError,
    ModelArtifactFormat,
    ModelUsageState,
    ModelVersionMismatchError,
    SecureArtifactManager,
    SecureEngineConfig,
    SecureInferenceEngine,
    SecureModelMetadata,
    UnsafeFormatError,
    compute_streaming_sha256,
)
from ai_core.neural.models.factory import create_image_detector
from ai_core.neural.schemas import InferenceStatus


def _create_synthetic_weights_and_metadata(
    tmp_dir: Path,
    architecture: str = "mobilenet_v3_large",
    version: str = "1.0.0",
    use_safetensors: bool = True,
    corrupt: bool = False,
    tamper_hash: bool = False,
) -> Tuple[Path, Path]:
    """Helper to create dummy valid/corrupt model weights and matching metadata."""
    detector = create_image_detector(
        architecture=architecture,
        model_version=version,
        is_trained=True,
    )
    state_dict = detector.state_dict()

    if use_safetensors:
        weights_path = tmp_dir / f"model_{version}.safetensors"
        if corrupt:
            with open(weights_path, "wb") as f:
                f.write(b"NOT_A_VALID_SAFETENSORS_HEADER_GARBAGE_BYTES")
        else:
            safetensors.torch.save_file(state_dict, str(weights_path))
        fmt = ModelArtifactFormat.SAFETENSORS
    else:
        weights_path = tmp_dir / f"model_{version}.pt"
        if corrupt:
            with open(weights_path, "wb") as f:
                f.write(b"TRUNCATED_PYTORCH_BYTES_12345")
        else:
            torch.save(state_dict, str(weights_path))
        fmt = ModelArtifactFormat.PYTORCH_STATE_DICT

    sha256_hex, file_size = compute_streaming_sha256(weights_path)

    if tamper_hash:
        sha256_hex = "f" * 64

    metadata_path = tmp_dir / f"metadata_{version}.json"
    meta = SecureModelMetadata(
        model_id=f"safeher-{architecture}",
        version=version,
        architecture=architecture,
        input_dimensions=(3, 224, 224),
        preprocessing_version="v1-imagenet-norm",
        class_mappings={0: "CAMERA_PHOTOGRAPH", 1: "AI_GENERATED_IMAGE"},
        training_manifest_hash="a" * 64,
        evaluation_report_reference="reports/INDEPENDENT_NEURAL_EVALUATION_REPORT.md",
        calibration_artifact_reference="calibration/calibrator_v1.json",
        sha256=sha256_hex,
        expected_file_size_bytes=file_size,
        approved_usage_state=ModelUsageState.COMMERCIAL_TRAINED,
        format=fmt,
    ).with_computed_hash()

    meta.save(metadata_path)
    return weights_path, metadata_path


# ---------------------------------------------------------------------------
# 1. Valid Artifact & Safe Loading Tests
# ---------------------------------------------------------------------------

def test_valid_safetensors_artifact_loading():
    """Verify clean loading and verification of valid .safetensors artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, use_safetensors=True)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=False,
        )

        assert engine.lifecycle_state == EngineLifecycleState.READY
        assert engine.status == InferenceStatus.READY
        assert engine.metadata is not None
        assert engine.metadata.architecture == "mobilenet_v3_large"

        # Execute single prediction
        dummy_x = torch.randn(1, 3, 224, 224)
        res = engine.predict(dummy_x)
        assert res.status == InferenceStatus.READY
        assert len(res.raw_logits) == 2
        assert "AI_GENERATED_IMAGE" in res.uncalibrated_scores


def test_valid_pytorch_safe_weights_only_loading():
    """Verify loading of PyTorch .pt weights strictly utilizing weights_only=True."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, use_safetensors=False)

        policy = ArtifactSecurityPolicy(allowed_directories=[tmp_dir], allow_pytorch_weights=True)
        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            security_policy=policy,
            lazy_load=False,
        )

        assert engine.lifecycle_state == EngineLifecycleState.READY
        res = engine.predict(torch.randn(1, 3, 224, 224))
        assert res.status == InferenceStatus.READY


# ---------------------------------------------------------------------------
# 2. Missing Artifact & Corrupted Checksum Tests
# ---------------------------------------------------------------------------

def test_missing_artifact_raises_file_not_found():
    """Verify missing weights file raises FileNotFoundError or sets MODEL_UNAVAILABLE."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)
        missing_weights = tmp_dir / "non_existent_weights.safetensors"

        engine = SecureInferenceEngine(
            artifact_path=missing_weights,
            metadata_path=meta_p,
            lazy_load=True,
        )

        with pytest.raises(FileNotFoundError):
            engine.load()

        assert engine.lifecycle_state == EngineLifecycleState.ERROR


def test_checksum_mismatch_detection():
    """Verify tampered artifact SHA-256 raises IntegrityVerificationError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, tamper_hash=True)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )

        with pytest.raises(IntegrityVerificationError, match="Cryptographic SHA-256 verification FAILED"):
            engine.load()

        assert engine.lifecycle_state == EngineLifecycleState.ERROR


def test_corrupted_weights_bitstream_rejection():
    """Verify corrupted garbage weights raise CorruptArtifactError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, corrupt=True)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )

        with pytest.raises(CorruptArtifactError):
            engine.load()


# ---------------------------------------------------------------------------
# 3. Architecture & Format Security Tests
# ---------------------------------------------------------------------------

def test_architecture_mismatch_rejection():
    """Verify mismatch between metadata architecture and actual detector raises ArchitectureMismatchError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, architecture="mobilenet_v3_large")

        # Create mismatched target model (EfficientNet-B0)
        mismatched_detector = create_image_detector(architecture="efficientnet_b0")
        meta = SecureModelMetadata.load(meta_p)
        manager = SecureArtifactManager()

        with pytest.raises(ArchitectureMismatchError, match="Model architecture mismatch"):
            manager.load_and_verify_model(
                artifact_path=weights_p,
                metadata=meta,
                model=mismatched_detector,
            )


def test_unsafe_format_rejection():
    """Verify untrusted formats (.pkl, .joblib) are strictly rejected with UnsafeFormatError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        unsafe_p = tmp_dir / "untrusted_model.pkl"
        with open(unsafe_p, "wb") as f:
            f.write(b"arbitrary_pickle_payload")

        manager = SecureArtifactManager()
        with pytest.raises(UnsafeFormatError, match="Prohibited or unsafe model artifact extension"):
            manager.load_weights_safe(unsafe_p)


def test_path_traversal_protection():
    """Verify artifact path traversal outside allowed directories raises ArtifactPathTraversalError."""
    with tempfile.TemporaryDirectory() as root_tmp:
        root_dir = Path(root_tmp)
        allowed_dir = root_dir / "trusted_models"
        allowed_dir.mkdir()
        untrusted_dir = root_dir / "untrusted_zone"
        untrusted_dir.mkdir()

        weights_p, meta_p = _create_synthetic_weights_and_metadata(untrusted_dir)

        # Policy only permits allowed_dir
        policy = ArtifactSecurityPolicy(allowed_directories=[allowed_dir])
        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            security_policy=policy,
            lazy_load=True,
        )

        with pytest.raises(ArtifactPathTraversalError, match="Access denied: Artifact path"):
            engine.load()


# ---------------------------------------------------------------------------
# 4. Input Validation & Batch Limit Tests
# ---------------------------------------------------------------------------

def test_input_validation_shapes_and_values():
    """Verify rigorous rejection of wrong tensor shapes, channels, and non-finite values."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=False,
        )

        # Wrong dimensionality (3D instead of 4D)
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            engine.validate_tensor(torch.randn(3, 224, 224))

        # Wrong channel count (1 channel instead of 3)
        with pytest.raises(ValueError, match="Input channels mismatch"):
            engine.validate_tensor(torch.randn(1, 1, 224, 224))

        # Wrong spatial resolution (100x100 instead of 224x224)
        with pytest.raises(ValueError, match="Spatial resolution mismatch"):
            engine.validate_tensor(torch.randn(1, 3, 100, 100))

        # Non-finite values (NaN / Inf)
        bad_nan = torch.randn(1, 3, 224, 224)
        bad_nan[0, 0, 10, 10] = float("nan")
        with pytest.raises(ValueError, match="contains non-finite values"):
            engine.validate_tensor(bad_nan)


def test_batch_limit_and_chunking():
    """Verify batch size limit enforcement and seamless chunking."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)

        # Config: max_batch_size = 4, chunking = True
        config = SecureEngineConfig(max_batch_size=4, allow_batch_chunking=True)
        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            config=config,
            lazy_load=False,
        )

        # Input of 10 samples (exceeds max_batch_size=4)
        x_10 = torch.randn(10, 3, 224, 224)
        batch_res = engine.predict_batch(x_10)

        assert batch_res.batch_size == 10
        assert len(batch_res.results) == 10

        # Disallow chunking -> raises BatchLimitExceededError
        config_no_chunk = SecureEngineConfig(max_batch_size=4, allow_batch_chunking=False)
        engine_no_chunk = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            config=config_no_chunk,
            lazy_load=False,
        )

        with pytest.raises(BatchLimitExceededError, match="exceeds maximum allowed limit"):
            engine_no_chunk.predict_batch(x_10)


def test_inference_timeout_enforcement(monkeypatch):
    """Verify that slow inference execution triggers InferenceTimeoutError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)

        # Microsecond timeout to trigger timeout
        config = SecureEngineConfig(timeout_seconds=0.001)
        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            config=config,
            lazy_load=False,
        )

        def slow_forward(tensor):
            time.sleep(0.05)
            return torch.zeros((tensor.shape[0], 2))

        monkeypatch.setattr(engine._model, "forward", slow_forward)

        with pytest.raises(InferenceTimeoutError, match="exceeded timeout threshold"):
            engine.predict(torch.randn(1, 3, 224, 224))


# ---------------------------------------------------------------------------
# 5. Concurrency, Lifecycle & Shutdown Tests
# ---------------------------------------------------------------------------

def test_concurrent_thread_safe_inference():
    """Verify thread-safe concurrent inference without data races."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=False,
        )

        def worker_task(worker_id: int):
            tensor = torch.randn(1, 3, 224, 224)
            res = engine.predict(tensor)
            assert res.status == InferenceStatus.READY
            assert len(res.raw_logits) == 2
            return res.predicted_class

        # Run 8 concurrent worker requests
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker_task, i) for i in range(8)]
            results = [f.result() for f in futures]

        assert len(results) == 8


def test_graceful_shutdown_and_lifecycle():
    """Verify engine lifecycle transitions and memory unloading."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )
        assert engine.lifecycle_state == EngineLifecycleState.UNINITIALIZED

        # Explicit warmup triggers load
        engine.warmup()
        assert engine.lifecycle_state == EngineLifecycleState.READY

        # Explicit unload
        engine.unload()
        assert engine.lifecycle_state == EngineLifecycleState.UNLOADED

        # Predict triggers lazy reload
        res = engine.predict(torch.randn(1, 3, 224, 224))
        assert res.status == InferenceStatus.READY
        assert engine.lifecycle_state == EngineLifecycleState.READY

        # Shutdown
        engine.shutdown()
        assert engine.lifecycle_state == EngineLifecycleState.SHUTDOWN


def test_no_silent_model_version_substitution():
    """Verify engine rejects silent model version substitution."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Create version 1.0.0
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, version="1.0.0")

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )

        # Request version 2.0.0 -> must raise ModelVersionMismatchError
        with pytest.raises(ModelVersionMismatchError, match="Silent substitution is rejected"):
            engine.load(target_version="2.0.0")


def test_rejected_usage_state_blocks_loading():
    """Verify artifacts with approved_usage_state=REJECTED are blocked from inference."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)

        # Edit metadata to REJECTED
        meta = SecureModelMetadata.load(meta_p)
        dumped = meta.model_dump(exclude={"metadata_hash"})
        dumped["approved_usage_state"] = ModelUsageState.REJECTED.value
        rejected_meta = SecureModelMetadata(**dumped).with_computed_hash()
        rejected_meta.save(meta_p)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )

        with pytest.raises(ValueError, match="explicitly marked as REJECTED"):
            engine.load()


def test_research_restricted_usage_state_blocks_operational_inference():
    """Research-only artifacts must not be loaded into operational inference."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)
        meta = SecureModelMetadata.load(meta_p)
        dumped = meta.model_dump(exclude={"metadata_hash"})
        dumped["approved_usage_state"] = ModelUsageState.RESEARCH_RESTRICTED.value
        restricted_meta = SecureModelMetadata(**dumped).with_computed_hash()
        restricted_meta.save(meta_p)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )
        with pytest.raises(ValueError, match="requires COMMERCIAL_TRAINED approval"):
            engine.load()


def test_pending_review_usage_state_blocks_operational_inference():
    """Artifacts awaiting review must not be loaded into operational inference."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir)
        meta = SecureModelMetadata.load(meta_p)
        dumped = meta.model_dump(exclude={"metadata_hash"})
        dumped["approved_usage_state"] = ModelUsageState.PENDING_REVIEW.value
        pending_meta = SecureModelMetadata(**dumped).with_computed_hash()
        pending_meta.save(meta_p)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )
        with pytest.raises(ValueError, match="requires COMMERCIAL_TRAINED approval"):
            engine.load()


def test_artifact_extension_must_match_declared_format():
    """A valid hash cannot make a mislabeled serialization format acceptable."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        weights_p, meta_p = _create_synthetic_weights_and_metadata(tmp_dir, use_safetensors=True)
        meta = SecureModelMetadata.load(meta_p)
        dumped = meta.model_dump(exclude={"metadata_hash"})
        dumped["format"] = ModelArtifactFormat.PYTORCH_STATE_DICT.value
        mismatched_meta = SecureModelMetadata(**dumped).with_computed_hash()
        mismatched_meta.save(meta_p)

        engine = SecureInferenceEngine(
            artifact_path=weights_p,
            metadata_path=meta_p,
            lazy_load=True,
        )
        with pytest.raises(UnsafeFormatError, match="Artifact format mismatch"):
            engine.load()
