"""Pydantic schemas and metadata contracts for SafeHER Milestone 4 Neural Detectors.

Defines typed configurations, class mappings, metadata contracts, and inference
result payloads with strict disclaimers preventing misinterpretation of uncalibrated
raw scores as calibrated probabilities or authenticity certifications.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class InferenceStatus(str, Enum):
    """Categorical operational status of neural inference."""

    READY = "READY"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNTRAINED_MODEL_REJECTED = "UNTRAINED_MODEL_REJECTED"
    INVALID_INPUT = "INVALID_INPUT"
    INTEGRITY_VERIFICATION_FAILED = "INTEGRITY_VERIFICATION_FAILED"
    ERROR = "ERROR"


class NeuralArchitecture(str, Enum):
    """Supported neural backbone architectures for SafeHER image detectors."""

    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    EFFICIENTNET_B0 = "efficientnet_b0"


# Explicit non-overlapping class mappings for Model A
MODEL_A_CLASS_MAPPING: Dict[int, str] = {
    0: "CAMERA_PHOTOGRAPH",
    1: "AI_GENERATED_IMAGE",
}

MODEL_A_LABEL_TO_INDEX: Dict[str, int] = {
    "CAMERA_PHOTOGRAPH": 0,
    "AI_GENERATED_IMAGE": 1,
}

# Supported domain definition
MODEL_A_SUPPORTED_DOMAIN = (
    "AI-generated synthetic image detection vs. authentic camera-origin photographs "
    "(RGB color format, 224x224 spatial resolution, normalized with standard ImageNet statistics)."
)

# Mandatory forensic disclaimer: negative detection != authenticity certification
MANDATORY_NEURAL_DISCLAIMER = (
    "SafeHER Neural Triage Disclaimer: Uncalibrated raw neural detector output for "
    "forensic investigation. Raw scores must NOT be interpreted as calibrated probabilities. "
    "Absence of detected synthetic artifacts does NOT certify that an image is authentic."
)


class InputConfiguration(BaseModel):
    """Input tensor specification and normalization contract for neural models."""

    model_config = ConfigDict(frozen=True)

    expected_channels: int = Field(
        default=3,
        description="Expected number of color channels (RGB=3).",
    )
    spatial_resolution: Tuple[int, int] = Field(
        default=(224, 224),
        description="Expected input height and width in pixels (H, W).",
    )
    mean: List[float] = Field(
        default=[0.485, 0.456, 0.406],
        description="Channel-wise normalization mean values.",
    )
    std: List[float] = Field(
        default=[0.229, 0.224, 0.225],
        description="Channel-wise normalization standard deviation values.",
    )
    color_space: str = Field(
        default="RGB",
        description="Expected color channel ordering.",
    )
    tensor_layout: str = Field(
        default="BCHW",
        description="Expected PyTorch tensor layout (Batch, Channel, Height, Width).",
    )


class ModelMetadata(BaseModel):
    """Immutable metadata descriptor embedded within each neural detector model."""

    model_config = ConfigDict(frozen=True)

    architecture: str = Field(
        ...,
        description="Identifier of the neural backbone architecture.",
    )
    model_version: str = Field(
        ...,
        description="Semantic version string of the model artifact.",
    )
    task_type: str = Field(
        default="AI_GENERATED_IMAGE_DETECTION",
        description="Forensic detection task scope.",
    )
    supported_domain: str = Field(
        default=MODEL_A_SUPPORTED_DOMAIN,
        description="Strictly defined operational domain for this detector.",
    )
    class_mapping: Dict[int, str] = Field(
        default=MODEL_A_CLASS_MAPPING,
        description="Explicit mapping from output logit index to human-readable class name.",
    )
    input_config: InputConfiguration = Field(
        default_factory=InputConfiguration,
        description="Expected input tensor shape, channels, and normalization.",
    )
    is_trained: bool = Field(
        default=False,
        description="Flag indicating whether model weights have undergone verified training.",
    )
    weights_sha256: Optional[str] = Field(
        default=None,
        description="Cryptographic SHA-256 digest of the serialized weights artifact.",
    )
    training_dataset_id: Optional[str] = Field(
        default=None,
        description="Identifier of the approved dataset manifest used for training.",
    )
    parameter_count: int = Field(
        default=0,
        description="Total number of trainable and non-trainable parameters.",
    )
    creation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of model export.",
    )


class ModelInferenceResult(BaseModel):
    """Structured inference payload returned by SafeHER neural detector services."""

    model_config = ConfigDict(frozen=True)

    raw_logits: List[float] = Field(
        ...,
        description="Raw unbounded logits output by the classification head.",
    )
    uncalibrated_scores: Dict[str, float] = Field(
        ...,
        description="Uncalibrated per-class softmax scores. MUST NOT be labeled as probabilities.",
    )
    predicted_class: str = Field(
        ...,
        description="Class name corresponding to the argmax logit.",
    )
    model_version: str = Field(
        ...,
        description="Semantic version string of the evaluating detector.",
    )
    architecture: str = Field(
        ...,
        description="Backbone architecture name.",
    )
    input_config: InputConfiguration = Field(
        ...,
        description="Input configuration applied during inference.",
    )
    status: InferenceStatus = Field(
        ...,
        description="Inference execution status.",
    )
    latency_ms: float = Field(
        ...,
        description="Inference duration in milliseconds on CPU.",
    )
    disclaimer: str = Field(
        default=MANDATORY_NEURAL_DISCLAIMER,
        description="Mandatory forensic disclaimer against authenticity certification.",
    )


class BatchInferenceResult(BaseModel):
    """Structured inference payload for batched neural predictions."""

    model_config = ConfigDict(frozen=True)

    results: List[ModelInferenceResult] = Field(
        ...,
        description="Per-sample inference results in the batch.",
    )
    batch_size: int = Field(
        ...,
        description="Number of samples evaluated in this batch.",
    )
    total_latency_ms: float = Field(
        ...,
        description="Total wall-clock latency for the batch in milliseconds.",
    )
    average_latency_ms: float = Field(
        ...,
        description="Average per-sample latency in milliseconds.",
    )
