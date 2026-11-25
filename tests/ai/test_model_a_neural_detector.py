"""Unit tests for SafeHER Model A Neural Image Detector.

Verifies:
- Configurable architecture factory and registry (MobileNetV3-Large & EfficientNet-B0).
- Strict input tensor validation (shape, channels, resolution, dtype, NaN/Inf).
- Explicit class mapping and model metadata exposure.
- Safe serialization (PyTorch & SafeTensors) and SHA-256 integrity verification.
- Rejection of missing, corrupted, or untrained model artifacts.
- Deterministic CPU inference configuration.
- Decoupled training harness and metric calculation.
- Forensic disclaimers preventing misinterpretation as calibrated probabilities or authenticity certifications.
"""

from pathlib import Path
import tempfile
from typing import Dict

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_core.neural.inference import (
    ModelAInferenceService,
    configure_deterministic_inference,
)
from ai_core.neural.models import (
    BaseNeuralDetector,
    EfficientNetB0Detector,
    MobileNetV3LargeDetector,
    compute_file_sha256,
    create_image_detector,
    list_supported_architectures,
    load_detector_checkpoint,
)
from ai_core.neural.schemas import (
    MANDATORY_NEURAL_DISCLAIMER,
    MODEL_A_CLASS_MAPPING,
    InferenceStatus,
    InputConfiguration,
    NeuralArchitecture,
)
from ai_core.neural.training import ModelATrainer, save_detector_checkpoint


# ============================================================================
# 1. Architecture Factory & Registry Tests
# ============================================================================

def test_list_supported_architectures():
    """Verify registry exposes supported backbones without hardcoding single choice."""
    archs = list_supported_architectures()
    assert "mobilenet_v3_large" in archs
    assert "efficientnet_b0" in archs
    assert len(archs) >= 2


def test_create_mobilenet_v3_detector():
    """Verify MobileNetV3-Large detector instantiation, layers, and metadata."""
    model = create_image_detector(NeuralArchitecture.MOBILENET_V3_LARGE)
    assert isinstance(model, MobileNetV3LargeDetector)
    assert isinstance(model, BaseNeuralDetector)
    assert model.parameter_count() > 1_500_000
    assert model.class_mapping == MODEL_A_CLASS_MAPPING

    metadata = model.get_metadata()
    assert metadata.architecture == "mobilenet_v3_large"
    assert metadata.class_mapping[0] == "CAMERA_PHOTOGRAPH"
    assert metadata.class_mapping[1] == "AI_GENERATED_IMAGE"
    assert metadata.is_trained is False


def test_create_efficientnet_b0_detector():
    """Verify EfficientNet-B0 detector instantiation, layers, and metadata."""
    model = create_image_detector(NeuralArchitecture.EFFICIENTNET_B0)
    assert isinstance(model, EfficientNetB0Detector)
    assert isinstance(model, BaseNeuralDetector)
    assert model.parameter_count() > 3_500_000
    assert model.class_mapping == MODEL_A_CLASS_MAPPING

    metadata = model.get_metadata()
    assert metadata.architecture == "efficientnet_b0"
    assert metadata.class_mapping[0] == "CAMERA_PHOTOGRAPH"
    assert metadata.class_mapping[1] == "AI_GENERATED_IMAGE"


def test_create_unsupported_architecture_raises():
    """Verify that requesting an unregistered architecture raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported neural architecture"):
        create_image_detector("resnet50_unapproved")


# ============================================================================
# 2. Input Tensor Validation Tests
# ============================================================================

@pytest.mark.parametrize("arch", ["mobilenet_v3_large", "efficientnet_b0"])
def test_input_validation_valid_tensor(arch: str):
    """Verify valid 4D float tensor passes without error."""
    model = create_image_detector(arch)
    valid_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    model.validate_input_tensor(valid_tensor)


def test_input_validation_reject_non_tensor():
    """Verify non-tensor input raises TypeError."""
    model = create_image_detector("mobilenet_v3_large")
    with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
        model.validate_input_tensor(np.zeros((1, 3, 224, 224)))


def test_input_validation_reject_integer_tensor():
    """Verify non-float tensor raises TypeError."""
    model = create_image_detector("mobilenet_v3_large")
    int_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.int64)
    with pytest.raises(TypeError, match="floating-point dtype"):
        model.validate_input_tensor(int_tensor)


def test_input_validation_reject_wrong_ndim():
    """Verify 2D, 3D, and 5D tensors raise ValueError."""
    model = create_image_detector("mobilenet_v3_large")
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model.validate_input_tensor(torch.randn(3, 224, 224))
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model.validate_input_tensor(torch.randn(1, 1, 3, 224, 224))


def test_input_validation_reject_empty_batch():
    """Verify empty batch dimension raises ValueError."""
    model = create_image_detector("mobilenet_v3_large")
    with pytest.raises(ValueError, match="Batch dimension must be >= 1"):
        model.validate_input_tensor(torch.randn(0, 3, 224, 224))


def test_input_validation_reject_wrong_channels():
    """Verify non-3 channel input (e.g. grayscale 1 channel or RGBA 4 channels) raises ValueError."""
    model = create_image_detector("mobilenet_v3_large")
    with pytest.raises(ValueError, match="Expected 3 input channels"):
        model.validate_input_tensor(torch.randn(1, 1, 224, 224))
    with pytest.raises(ValueError, match="Expected 3 input channels"):
        model.validate_input_tensor(torch.randn(1, 4, 224, 224))


def test_input_validation_reject_wrong_resolution():
    """Verify incorrect spatial resolution raises ValueError."""
    model = create_image_detector("mobilenet_v3_large")
    with pytest.raises(ValueError, match=r"Expected spatial resolution \(224, 224\)"):
        model.validate_input_tensor(torch.randn(1, 3, 256, 256))
    with pytest.raises(ValueError, match=r"Expected spatial resolution \(224, 224\)"):
        model.validate_input_tensor(torch.randn(1, 3, 128, 128))


def test_input_validation_reject_nan_and_inf():
    """Verify input containing NaN or Inf values raises ValueError."""
    model = create_image_detector("mobilenet_v3_large")
    nan_tensor = torch.randn(1, 3, 224, 224)
    nan_tensor[0, 0, 10, 10] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.validate_input_tensor(nan_tensor)

    inf_tensor = torch.randn(1, 3, 224, 224)
    inf_tensor[0, 0, 10, 10] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.validate_input_tensor(inf_tensor)


# ============================================================================
# 3. Forward Pass & Feature Extraction Tests
# ============================================================================

@pytest.mark.parametrize("arch", ["mobilenet_v3_large", "efficientnet_b0"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_forward_pass_logits_shape(arch: str, batch_size: int):
    """Verify forward pass returns raw logits tensor of shape (B, 2)."""
    model = create_image_detector(arch)
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, 2)
    assert logits.dtype == torch.float32


def test_feature_extraction_dims():
    """Verify feature extractor dimensions match backbone specifications."""
    m_mbv3 = create_image_detector("mobilenet_v3_large")
    m_eff = create_image_detector("efficientnet_b0")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feat_mbv3 = m_mbv3.extract_features(x)
        feat_eff = m_eff.extract_features(x)

    assert feat_mbv3.shape == (2, 960)
    assert feat_eff.shape == (2, 1280)


# ============================================================================
# 4. Deterministic Inference Verification
# ============================================================================

def test_deterministic_cpu_inference():
    """Verify deterministic CPU execution produces bitwise identical logits."""
    configure_deterministic_inference(num_threads=1, seed=123)
    model = create_image_detector("mobilenet_v3_large")
    model.eval()

    torch.manual_seed(999)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


# ============================================================================
# 5. Checkpointing & Integrity Verification Tests
# ============================================================================

def test_checkpoint_save_and_load_pt():
    """Verify PyTorch .pt checkpoint export with SHA-256 and loading."""
    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        saved_path, sha256 = save_detector_checkpoint(
            model,
            ckpt_path,
            extra_metadata={
                "is_trained": True,
                "training_completed": True,
                "epoch": 3,
                "training_manifest_hash": "a" * 64,
                "split_manifest_hash": "b" * 64,
                "dataset_versions": ["fixture-v1"],
                "approved_usage_state": "COMMERCIAL_TRAINED",
            },
        )

        assert saved_path.is_file()
        assert len(sha256) == 64
        assert compute_file_sha256(saved_path) == sha256

        # Load into new model
        target_model = create_image_detector("mobilenet_v3_large")
        load_detector_checkpoint(target_model, saved_path, expected_sha256=sha256)
        assert target_model.is_trained is True
        assert target_model.weights_sha256 == sha256


def test_checkpoint_sha256_mismatch_rejected():
    """Verify checkpoint loading fails if computed SHA-256 does not match expected."""
    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        save_detector_checkpoint(model, ckpt_path)

        target_model = create_image_detector("mobilenet_v3_large")
        wrong_hash = "0" * 64
        with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
            load_detector_checkpoint(target_model, ckpt_path, expected_sha256=wrong_hash)


@pytest.mark.parametrize("usage_state", [None, "PENDING_REVIEW", "RESEARCH_RESTRICTED", "REJECTED"])
def test_checkpoint_without_commercial_usage_approval_is_rejected(usage_state):
    """Only checkpoints explicitly cleared for commercial training may be loaded for app inference."""
    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    metadata = {
        "is_trained": True,
        "training_completed": True,
        "epoch": 3,
        "training_manifest_hash": "a" * 64,
        "split_manifest_hash": "b" * 64,
        "dataset_versions": ["fixture-v1"],
    }
    if usage_state is not None:
        metadata["approved_usage_state"] = usage_state

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "restricted_model.pt"
        save_detector_checkpoint(model, checkpoint_path, extra_metadata=metadata)
        target_model = create_image_detector("mobilenet_v3_large")
        with pytest.raises(ValueError, match="ARTIFACT_NOT_APPROVED"):
            load_detector_checkpoint(target_model, checkpoint_path, enforce_trained=True)


def test_checkpoint_missing_file_raises():
    """Verify loading non-existent checkpoint raises FileNotFoundError."""
    target_model = create_image_detector("mobilenet_v3_large")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_detector_checkpoint(target_model, "non_existent_weights.pt")


# ============================================================================
# 6. Untrained & Missing Model Rejection Tests
# ============================================================================

def test_inference_service_rejects_missing_weights():
    """Verify service reports MODEL_UNAVAILABLE and never fabricates scores when weights are missing."""
    svc = ModelAInferenceService(checkpoint_path="non_existent_weights.pt", enforce_trained=True)
    assert svc.status == InferenceStatus.MODEL_UNAVAILABLE

    x = torch.randn(1, 3, 224, 224)
    res = svc.predict(x)
    assert res.status == InferenceStatus.MODEL_UNAVAILABLE
    assert res.raw_logits == []
    assert res.uncalibrated_scores == {}
    assert res.predicted_class == "UNKNOWN"
    assert MANDATORY_NEURAL_DISCLAIMER in res.disclaimer


def test_inference_service_rejects_untrained_model():
    """Verify service rejects explicitly untrained model artifacts."""
    untrained_model = create_image_detector("mobilenet_v3_large", is_trained=False)
    svc = ModelAInferenceService(model=untrained_model, enforce_trained=True)
    assert svc.status == InferenceStatus.UNTRAINED_MODEL_REJECTED

    x = torch.randn(1, 3, 224, 224)
    res = svc.predict(x)
    assert res.status == InferenceStatus.UNTRAINED_MODEL_REJECTED
    assert res.raw_logits == []
    assert res.uncalibrated_scores == {}
    assert res.predicted_class == "UNKNOWN"


def test_load_untrained_checkpoint_flag_rejected():
    """Verify load_detector_checkpoint rejects checkpoints flagged with is_trained=False."""
    model = create_image_detector("mobilenet_v3_large", is_trained=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "untrained.pt"
        save_detector_checkpoint(model, ckpt_path, extra_metadata={"is_trained": False})

        target_model = create_image_detector("mobilenet_v3_large")
        with pytest.raises(ValueError, match="explicitly flagged as UNTRAINED"):
            load_detector_checkpoint(target_model, ckpt_path, enforce_trained=True)


# ============================================================================
# 7. Inference Service Execution & Payload Tests
# ============================================================================

def test_inference_service_prediction_payload():
    """Verify complete output contract: raw logits, uncalibrated scores, status, and disclaimer."""
    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    svc = ModelAInferenceService(model=model, enforce_trained=True)
    assert svc.status == InferenceStatus.READY

    x = torch.randn(1, 3, 224, 224)
    res = svc.predict(x)

    assert res.status == InferenceStatus.READY
    assert len(res.raw_logits) == 2
    assert "CAMERA_PHOTOGRAPH" in res.uncalibrated_scores
    assert "AI_GENERATED_IMAGE" in res.uncalibrated_scores
    # Uncalibrated softmax scores sum to ~1.0
    total_score = sum(res.uncalibrated_scores.values())
    assert abs(total_score - 1.0) < 1e-4

    assert res.predicted_class in ["CAMERA_PHOTOGRAPH", "AI_GENERATED_IMAGE"]
    assert res.model_version == "1.0.0-m4.3"
    assert res.architecture == "mobilenet_v3_large"
    assert res.latency_ms >= 0.0

    # Ensure SafeHER never claims an image is authentic
    assert "AUTHENTIC" not in res.predicted_class
    assert "does NOT certify that an image is authentic" in res.disclaimer


def test_inference_service_batch_prediction():
    """Verify batched inference execution returns structured per-sample results."""
    model = create_image_detector("mobilenet_v3_large", is_trained=True)
    svc = ModelAInferenceService(model=model, enforce_trained=True)

    batch_x = torch.randn(3, 3, 224, 224)
    batch_res = svc.predict_batch(batch_x)

    assert batch_res.batch_size == 3
    assert len(batch_res.results) == 3
    assert batch_res.total_latency_ms >= 0.0
    for r in batch_res.results:
        assert r.status == InferenceStatus.READY
        assert len(r.raw_logits) == 2
        assert len(r.uncalibrated_scores) == 2


def test_image_preprocessing_pipeline():
    """Verify PIL and numpy image preprocessing formats and normalization."""
    svc = ModelAInferenceService(enforce_trained=False)

    # From PIL Image
    pil_img = Image.new("RGB", (300, 400), color=(128, 64, 32))
    tensor_from_pil = svc.preprocess_image(pil_img)
    assert tensor_from_pil.shape == (1, 3, 224, 224)
    assert tensor_from_pil.dtype == torch.float32

    # From numpy array (uint8)
    np_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    tensor_from_np = svc.preprocess_image(np_img)
    assert tensor_from_np.shape == (1, 3, 224, 224)
    assert tensor_from_np.dtype == torch.float32


# ============================================================================
# 8. Decoupled Training Harness Tests
# ============================================================================

def test_training_harness_decoupled_execution():
    """Verify decoupled ModelATrainer executes training, tracks metrics, and exports trained model."""
    model = create_image_detector("mobilenet_v3_large")
    assert model.is_trained is False

    # Synthetic training batch
    x_train = torch.randn(4, 3, 224, 224)
    y_train = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2)

    trainer = ModelATrainer(model=model, learning_rate=1e-3, seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        history = trainer.train(train_loader, epochs=1, checkpoint_dir=tmpdir)

        assert len(history) == 1
        assert history[0].sample_count == 4
        assert history[0].loss > 0.0
        assert model.is_trained is True

        exported_ckpt = Path(tmpdir) / "mobilenet_v3_large_checkpoint.pt"
        assert exported_ckpt.is_file()

        # This convenience trainer has no approved dataset/split provenance.
        # Its checkpoint must therefore remain ineligible for production inference.
        loaded_model = create_image_detector("mobilenet_v3_large")
        with pytest.raises(ValueError, match="training_completed=true"):
            load_detector_checkpoint(loaded_model, exported_ckpt, enforce_trained=True)

        svc = ModelAInferenceService(model=loaded_model, enforce_trained=True)
        res = svc.predict(torch.randn(1, 3, 224, 224))
        assert res.status == InferenceStatus.UNTRAINED_MODEL_REJECTED
        assert res.raw_logits == []
        assert res.uncalibrated_scores == {}
