"""Unit and integration tests for SafeHER Model Rollback and Operational Recovery (M4).

Covers all 8 mandated operational failure and recovery modes:
1. Missing artifact rejection.
2. Checksum mismatch detection & rejection.
3. Incompatible metadata & unapproved usage state rejection.
4. Failed activation and zero-downtime fallback.
5. Interrupted deployment & prevention of partial artifact activation.
6. Deterministic rollback to previous stable model & post-deployment guard.
7. Worker restart recovery and corrupt state file handling.
8. Concurrent multi-threaded inference during active version switching.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import tempfile
import threading
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
    CandidateModelDeployment,
    CorruptArtifactError,
    DeploymentActivationError,
    DeploymentTrigger,
    IntegrityVerificationError,
    ModelArtifactFormat,
    ModelRollbackRecoveryController,
    ModelUsageState,
    ModelVersionMismatchError,
    ModelVersionTransition,
    RollbackUnavailableError,
    SecureEngineConfig,
    SecureInferenceEngine,
    SecureModelMetadata,
    TransitionStatus,
    UnapprovedUsageStateError,
    compute_streaming_sha256,
)
from ai_core.neural.models.factory import create_image_detector
from ai_core.neural.schemas import InferenceStatus


def _create_test_model_artifacts(
    tmp_dir: Path,
    version: str = "1.0.0",
    architecture: str = "mobilenet_v3_large",
    approved_state: ModelUsageState = ModelUsageState.COMMERCIAL_TRAINED,
    corrupt_weights: bool = False,
    tamper_hash: bool = False,
    file_prefix: str = "model",
) -> Tuple[Path, Path]:
    """Helper to generate valid or adversarial synthetic model weights and metadata."""
    detector = create_image_detector(
        architecture=architecture,
        model_version=version,
        is_trained=True,
    )
    state_dict = detector.state_dict()

    weights_path = tmp_dir / f"{file_prefix}_{version}.safetensors"
    if corrupt_weights:
        with open(weights_path, "wb") as f:
            f.write(b"CORRUPT_SAFETENSORS_GARBAGE_PAYLOAD_BYTES_XYZ")
    else:
        safetensors.torch.save_file(state_dict, str(weights_path))

    sha256_hex, file_size = compute_streaming_sha256(weights_path)
    if tamper_hash:
        sha256_hex = "0" * 64

    metadata_path = tmp_dir / f"metadata_{file_prefix}_{version}.json"
    meta = SecureModelMetadata(
        model_id=f"safeher-{architecture}",
        version=version,
        architecture=architecture,
        input_dimensions=(3, 224, 224),
        preprocessing_version="v1-imagenet-norm",
        class_mappings={0: "CAMERA_PHOTOGRAPH", 1: "AI_GENERATED_IMAGE"},
        training_manifest_hash="d" * 64,
        evaluation_report_reference=f"reports/eval_{version}.md",
        calibration_artifact_reference=f"calibration/cal_{version}.json",
        sha256=sha256_hex,
        expected_file_size_bytes=file_size,
        approved_usage_state=approved_state,
        format=ModelArtifactFormat.SAFETENSORS,
    ).with_computed_hash()

    meta.save(metadata_path)
    return weights_path, metadata_path


# ---------------------------------------------------------------------------
# 1. Missing Artifact Tests
# ---------------------------------------------------------------------------

def test_missing_artifact_rejected():
    """Verify missing weights or metadata path raises FileNotFoundError and active model is intact."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")

        controller = ModelRollbackRecoveryController()
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        controller.deploy_candidate(c_v1)
        assert controller.active_version == "1.0.0"

        # Missing weights file
        c_missing_w = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=tmp_dir / "non_existent_weights.safetensors",
            metadata_path=m_v1,
        )
        with pytest.raises(FileNotFoundError, match="Candidate model weights not found"):
            controller.deploy_candidate(c_missing_w)

        # Missing metadata file
        c_missing_m = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v1,
            metadata_path=tmp_dir / "non_existent_metadata.json",
        )
        with pytest.raises(FileNotFoundError, match="Candidate model metadata not found"):
            controller.deploy_candidate(c_missing_m)

        # Active model must remain 1.0.0 with transition history recorded
        assert controller.active_version == "1.0.0"
        history = controller.transition_history
        assert len(history) == 3  # 1 initial success + 2 failed validations
        assert history[-1].status == TransitionStatus.FAILED_VALIDATION
        assert history[-1].to_version == "2.0.0"


# ---------------------------------------------------------------------------
# 2. Checksum Mismatch Detection Tests
# ---------------------------------------------------------------------------

def test_checksum_mismatch_rejected():
    """Verify tampered weights or tampered metadata hash raises IntegrityVerificationError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")
        w_v2, m_v2 = _create_test_model_artifacts(tmp_dir, version="2.0.0", tamper_hash=True)

        controller = ModelRollbackRecoveryController()
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        controller.deploy_candidate(c_v1)

        c_v2_tampered = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v2,
            metadata_path=m_v2,
        )

        with pytest.raises(IntegrityVerificationError, match="Candidate weights SHA-256 mismatch"):
            controller.deploy_candidate(c_v2_tampered)

        # Verify active version remains v1.0.0
        assert controller.active_version == "1.0.0"
        assert controller.transition_history[-1].status == TransitionStatus.FAILED_VALIDATION


def test_candidate_expected_sha256_mismatch():
    """Verify candidate deployment fails if expected_sha256 specification doesn't match artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")

        controller = ModelRollbackRecoveryController()
        c_mismatch = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
            expected_sha256="1" * 64,  # Incorrect hash specification
        )

        with pytest.raises(IntegrityVerificationError, match=r"do not match expected deploy hash"):
            controller.deploy_candidate(c_mismatch)

        assert controller.active_version is None


# ---------------------------------------------------------------------------
# 3. Incompatible Metadata & Usage State Rejection Tests
# ---------------------------------------------------------------------------

def test_incompatible_metadata_rejection():
    """Verify version mismatch and architecture mismatch raise specific exceptions."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")

        controller = ModelRollbackRecoveryController()

        # Version mismatch (anti-pattern: no silent substitution)
        c_ver_mismatch = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",  # Metadata has 1.0.0
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        with pytest.raises(ModelVersionMismatchError, match="Silent substitution is rejected"):
            controller.deploy_candidate(c_ver_mismatch)

        # Architecture mismatch
        c_arch_mismatch = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
            expected_architecture="resnet50",  # Metadata has mobilenet_v3_large
        )
        with pytest.raises(ArchitectureMismatchError, match="Architecture mismatch"):
            controller.deploy_candidate(c_arch_mismatch)


def test_unapproved_experimental_state_rejected():
    """Anti-pattern enforcement: Verify experimental or research models are NEVER enabled automatically."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_exp, m_exp = _create_test_model_artifacts(
            tmp_dir,
            version="1.5.0-exp",
            approved_state=ModelUsageState.RESEARCH_RESTRICTED,
        )

        controller = ModelRollbackRecoveryController()
        c_exp = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.5.0-exp",
            artifact_path=w_exp,
            metadata_path=m_exp,
        )

        with pytest.raises(UnapprovedUsageStateError, match="SafeHER operational policy strictly requires 'COMMERCIAL_TRAINED'"):
            controller.deploy_candidate(c_exp)

        assert controller.active_version is None
        assert controller.transition_history[-1].status == TransitionStatus.FAILED_VALIDATION


# ---------------------------------------------------------------------------
# 4. Failed Activation Tests
# ---------------------------------------------------------------------------

def test_failed_activation_aborts_cleanly():
    """Verify that corrupt weights fail in-memory warmup and leave active model operational."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")
        w_corrupt, m_corrupt = _create_test_model_artifacts(
            tmp_dir,
            version="2.0.0",
            corrupt_weights=True,
        )

        controller = ModelRollbackRecoveryController()
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        controller.deploy_candidate(c_v1)
        assert controller.active_version == "1.0.0"

        c_corrupt = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_corrupt,
            metadata_path=m_corrupt,
        )

        with pytest.raises(DeploymentActivationError, match="Candidate activation failed"):
            controller.deploy_candidate(c_corrupt)

        # Active model remains v1.0.0 and operational
        assert controller.active_version == "1.0.0"
        dummy_input = torch.randn(1, 3, 224, 224)
        res = controller.predict(dummy_input)
        assert res.model_version == "1.0.0"
        assert res.status == InferenceStatus.READY

        # Audit history records FAILED_ACTIVATION
        assert controller.transition_history[-1].status == TransitionStatus.FAILED_ACTIVATION
        assert controller.transition_history[-1].to_version == "2.0.0"


# ---------------------------------------------------------------------------
# 5. Interrupted Deployment & Companion Model Atomic Binding Tests
# ---------------------------------------------------------------------------

def test_interrupted_deployment_prevents_partial_activation():
    """Verify that companion model validation/activation failure aborts entire composite deployment."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0", file_prefix="primary")
        w_v2, m_v2 = _create_test_model_artifacts(tmp_dir, version="2.0.0", file_prefix="primary")
        w_comp_good, m_comp_good = _create_test_model_artifacts(tmp_dir, version="1.0.0", file_prefix="comp")
        w_comp_bad, m_comp_bad = _create_test_model_artifacts(
            tmp_dir, version="2.0.0", file_prefix="comp", corrupt_weights=True
        )

        controller = ModelRollbackRecoveryController()

        # Initial baseline deployment
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
            companion_models={"facial_detector": (w_comp_good, m_comp_good)},
        )
        controller.deploy_candidate(c_v1)
        assert controller.active_version == "1.0.0"

        # Candidate v2 with broken companion model
        c_v2_broken_comp = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v2,
            metadata_path=m_v2,
            companion_models={"facial_detector": (w_comp_bad, m_comp_bad)},
        )

        with pytest.raises(DeploymentActivationError):
            controller.deploy_candidate(c_v2_broken_comp)

        # Neither primary nor companion was updated to v2.0.0 (zero partial activation)
        assert controller.active_version == "1.0.0"
        assert controller.transition_history[-1].status == TransitionStatus.FAILED_ACTIVATION


# ---------------------------------------------------------------------------
# 6. Rollback & Post-Deployment Verification Tests
# ---------------------------------------------------------------------------

def test_rollback_restores_previous_stable_model():
    """Verify that manual rollback smoothly switches from v2.0.0 back to previous stable v1.0.0."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")
        w_v2, m_v2 = _create_test_model_artifacts(tmp_dir, version="2.0.0")

        controller = ModelRollbackRecoveryController()

        # 1. Deploy v1.0.0
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        controller.deploy_candidate(c_v1)
        assert controller.active_version == "1.0.0"
        assert controller.previous_stable_version is None

        # 2. Upgrade to v2.0.0
        c_v2 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v2,
            metadata_path=m_v2,
        )
        controller.deploy_candidate(c_v2)
        assert controller.active_version == "2.0.0"
        assert controller.previous_stable_version == "1.0.0"

        # 3. Execute rollback
        trans = controller.rollback(reason="Model B regression in production")
        assert trans.status == TransitionStatus.ROLLED_BACK
        assert trans.from_version == "2.0.0"
        assert trans.to_version == "1.0.0"
        assert controller.active_version == "1.0.0"

        # Prediction verifies engine is running v1.0.0
        res = controller.predict(torch.randn(1, 3, 224, 224))
        assert res.model_version == "1.0.0"


def test_rollback_unavailable_error():
    """Verify attempting to rollback without a previously active model raises RollbackUnavailableError."""
    controller = ModelRollbackRecoveryController()
    with pytest.raises(RollbackUnavailableError, match="no previous stable model version is registered"):
        controller.rollback()


def test_post_deployment_corruption_triggers_rollback():
    """Verify disk file tampering immediately after activation triggers automatic rollback."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")
        w_v2, m_v2 = _create_test_model_artifacts(tmp_dir, version="2.0.0")

        controller = ModelRollbackRecoveryController()
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        controller.deploy_candidate(c_v1)
        assert controller.active_version == "1.0.0"

        # Simulate race condition/corruption right during post-verify by hooking compute_streaming_sha256
        c_v2 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v2,
            metadata_path=m_v2,
        )

        # Corrupt file right before post-deployment check by modifying the candidate file on disk
        # We can simulate this by deploying candidate with a wrapper or replacing the file after warmup
        orig_warmup = SecureInferenceEngine.warmup

        def corrupting_warmup(self, sample_count=1):
            orig_warmup(self, sample_count=sample_count)
            # Truncate weights file on disk to simulate bit rot / disk write race
            with open(w_v2, "wb") as f:
                f.write(b"CORRUPTED_POST_DEPLOYMENT")

        try:
            SecureInferenceEngine.warmup = corrupting_warmup
            with pytest.raises(IntegrityVerificationError, match="Post-deployment checksum mismatch"):
                controller.deploy_candidate(c_v2)

            # Assert automatic rollback reverted back to v1.0.0
            assert controller.active_version == "1.0.0"
            res = controller.predict(torch.randn(1, 3, 224, 224))
            assert res.model_version == "1.0.0"

            # Transition history must show ROLLED_BACK
            rolled_back_trans = [t for t in controller.transition_history if t.status == TransitionStatus.ROLLED_BACK]
            assert len(rolled_back_trans) >= 1
        finally:
            SecureInferenceEngine.warmup = orig_warmup


# ---------------------------------------------------------------------------
# 7. Worker Restart Recovery Tests
# ---------------------------------------------------------------------------

def test_worker_restart_recovery_and_corrupt_state_handling():
    """Verify worker restart recovers persisted active model or safely falls back on corrupt state."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        state_file = tmp_dir / "active_model_state.json"
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")
        w_v2, m_v2 = _create_test_model_artifacts(tmp_dir, version="2.0.0")

        # 1. Controller A deploys v2.0.0 and saves state
        controller_a = ModelRollbackRecoveryController(state_persistence_path=state_file)
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        c_v2 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v2,
            metadata_path=m_v2,
        )
        controller_a.deploy_candidate(c_v1)
        controller_a.deploy_candidate(c_v2)
        assert controller_a.active_version == "2.0.0"
        assert state_file.is_file()

        # 2. Worker restart: Controller B starts up with persisted state
        controller_b = ModelRollbackRecoveryController(state_persistence_path=state_file)
        assert controller_b.active_version is None
        trans = controller_b.recover_on_worker_restart(fallback_candidate=c_v1)
        assert trans is not None
        assert controller_b.active_version == "2.0.0"
        assert controller_b.predict(torch.randn(1, 3, 224, 224)).model_version == "2.0.0"

        # 3. Simulate state file corruption: Controller C starts up
        with open(state_file, "w", encoding="utf-8") as f:
            f.write("{ INVALID_JSON_DATA_CORRUPTED ...")

        controller_c = ModelRollbackRecoveryController(state_persistence_path=state_file)
        trans_c = controller_c.recover_on_worker_restart(fallback_candidate=c_v1)
        # Should gracefully fall back to baseline c_v1 without crashing
        assert trans_c is not None
        assert controller_c.active_version == "1.0.0"
        assert controller_c.predict(torch.randn(1, 3, 224, 224)).model_version == "1.0.0"


# ---------------------------------------------------------------------------
# 8. Concurrent Inference During Version Switching Tests
# ---------------------------------------------------------------------------

def test_concurrent_inference_during_version_switching():
    """Verify zero dropped requests or race conditions during rapid concurrent deployment & rollback."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        w_v1, m_v1 = _create_test_model_artifacts(tmp_dir, version="1.0.0")
        w_v2, m_v2 = _create_test_model_artifacts(tmp_dir, version="2.0.0")

        controller = ModelRollbackRecoveryController()
        c_v1 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="1.0.0",
            artifact_path=w_v1,
            metadata_path=m_v1,
        )
        c_v2 = CandidateModelDeployment(
            model_id="safeher-mobilenet",
            target_version="2.0.0",
            artifact_path=w_v2,
            metadata_path=m_v2,
        )

        controller.deploy_candidate(c_v1)
        assert controller.active_version == "1.0.0"

        stop_event = threading.Event()
        inference_results = []
        inference_errors = []

        def worker_loop(worker_id: int):
            dummy_input = torch.randn(1, 3, 224, 224)
            while not stop_event.is_set():
                try:
                    res = controller.predict(dummy_input)
                    inference_results.append((worker_id, res.model_version, res.status))
                except Exception as ex:
                    inference_errors.append((worker_id, str(ex)))
                time.sleep(0.005)

        # Launch 6 concurrent inference workers
        workers = [threading.Thread(target=worker_loop, args=(i,), daemon=True) for i in range(6)]
        for w in workers:
            w.start()

        time.sleep(0.05)

        # Concurrently perform version upgrade
        trans_upgrade = controller.deploy_candidate(c_v2)
        assert trans_upgrade.status == TransitionStatus.SUCCESS
        assert controller.active_version == "2.0.0"

        time.sleep(0.05)

        # Concurrently perform rollback
        trans_rollback = controller.rollback(reason="Load test verification rollback")
        assert trans_rollback.status == TransitionStatus.ROLLED_BACK
        assert controller.active_version == "1.0.0"

        time.sleep(0.05)

        # Stop workers
        stop_event.set()
        for w in workers:
            w.join(timeout=2.0)

        # Assert zero errors during concurrent transitions
        assert len(inference_errors) == 0, f"Concurrent inferences encountered errors: {inference_errors}"
        assert len(inference_results) >= 5, "Expected at least 5 concurrent inference calls to succeed"

        versions_served = {r[1] for r in inference_results}
        # Inferences must have completed against either 1.0.0 or 2.0.0
        assert "1.0.0" in versions_served
        for wid, ver, st in inference_results:
            assert st == InferenceStatus.READY
            assert ver in ("1.0.0", "2.0.0")
