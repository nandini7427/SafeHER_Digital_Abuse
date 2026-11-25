"""Adversarial robustness and transformation testing framework for SafeHER Image Intelligence.

Evaluates detector behavior under 12 realistic image transformations:
1. JPEG compression
2. PNG conversion
3. Resizing (downscale/upscale)
4. Cropping
5. Blur
6. Brightness changes
7. Contrast changes
8. Color transformations (grayscale/tint)
9. Screenshot conversion
10. Metadata removal
11. Re-encoding (multi-pass re-save)
12. Minor image noise (additive Gaussian)

Strictly preserves original evidence bytes, generates isolated derivatives,
tracks SHA-256 cryptographic provenance, measures score drift, and detects
false positive/negative transitions and uncertainty escalations.
"""

from dataclasses import dataclass, field
import hashlib
import io
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ai_core.image_intelligence.detector import BaseImageDetector, SafeHERImageDetector
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ImageAssessmentResponse,
    UncertaintyState,
)


@dataclass(frozen=True)
class RobustnessDerivative:
    """Isolated test derivative generated from pristine evidence."""

    original_sample_id: str
    transformation_name: str
    original_bytes: bytes
    original_sha256: str
    derivative_bytes: bytes
    derivative_sha256: str
    transformation_params: Dict[str, Any]
    is_manipulated: bool


@dataclass(frozen=True)
class TransformationPairEvaluation:
    """Side-by-side comparison between clean original and perturbed derivative."""

    sample_id: str
    transformation_name: str
    is_manipulated: bool
    clean_score: float
    perturbed_score: float
    score_delta: float  # perturbed_score - clean_score
    clean_label: AssessmentLabel
    perturbed_label: AssessmentLabel
    label_changed: bool
    is_false_positive_transition: bool
    is_false_negative_transition: bool
    clean_uncertainty: UncertaintyState
    perturbed_uncertainty: UncertaintyState
    uncertainty_escalated: bool
    quality_warnings_triggered: List[str]
    model_failed: bool


@dataclass(frozen=True)
class TransformationRobustnessSummary:
    """Aggregate robustness metrics for a specific transformation type."""

    transformation_name: str
    sample_count: int
    mean_score_drift: float
    max_score_drift: float
    score_std_dev: float
    label_flip_rate: float
    false_positive_transitions: int
    false_negative_transitions: int
    uncertainty_escalation_rate: float
    model_failure_count: int
    quality_warning_trigger_rate: float
    observations: str


# ---------------------------------------------------------------------------
# 12 Transformation Generators (Pristine Original Preserved)
# ---------------------------------------------------------------------------

def apply_jpeg_compression(image_bytes: bytes, quality: int = 50) -> Tuple[bytes, Dict[str, Any]]:
    """Apply lossy JPEG re-compression."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), {"quality": quality}


def apply_png_conversion(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Convert losslessly to PNG format."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), {"format": "PNG"}


def apply_resizing(image_bytes: bytes, scale_factor: float = 0.5) -> Tuple[bytes, Dict[str, Any]]:
    """Downscale and re-upscale back to original dimensions (simulates messaging preview)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    down_w = max(16, int(w * scale_factor))
    down_h = max(16, int(h * scale_factor))
    downscaled = img.resize((down_w, down_h), resample=Image.Resampling.BILINEAR)
    upscaled = downscaled.resize((w, h), resample=Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    upscaled.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"scale_factor": scale_factor, "resample": "BILINEAR"}


def apply_cropping(image_bytes: bytes, crop_ratio: float = 0.8) -> Tuple[bytes, Dict[str, Any]]:
    """Center crop and resize back to original dimensions."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    cw = int(w * crop_ratio)
    ch = int(h * crop_ratio)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
    restored = cropped.resize((w, h), resample=Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    restored.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"crop_ratio": crop_ratio, "crop_box": (x0, y0, x0 + cw, y0 + ch)}


def apply_blur(image_bytes: bytes, radius: float = 2.5) -> Tuple[bytes, Dict[str, Any]]:
    """Apply Gaussian blur."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"blur_type": "gaussian", "radius": radius}


def apply_brightness_change(image_bytes: bytes, factor: float = 1.25) -> Tuple[bytes, Dict[str, Any]]:
    """Shift pixel brightness."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    enhanced = enhancer.enhance(factor)
    buf = io.BytesIO()
    enhanced.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"brightness_factor": factor}


def apply_contrast_change(image_bytes: bytes, factor: float = 0.65) -> Tuple[bytes, Dict[str, Any]]:
    """Shift dynamic contrast."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    enhancer = ImageEnhance.Contrast(img)
    enhanced = enhancer.enhance(factor)
    buf = io.BytesIO()
    enhanced.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"contrast_factor": factor}


def apply_color_transformation(image_bytes: bytes, mode: str = "grayscale") -> Tuple[bytes, Dict[str, Any]]:
    """Convert to grayscale or apply subtle color tint."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if mode == "grayscale":
        # Desaturate to 3-channel grayscale
        gray = img.convert("L").convert("RGB")
    else:
        # Sepia tint
        arr = np.array(img, dtype=np.float32)
        sepia = np.zeros_like(arr)
        sepia[:, :, 0] = np.clip(0.393 * arr[:, :, 0] + 0.769 * arr[:, :, 1] + 0.189 * arr[:, :, 2], 0, 255)
        sepia[:, :, 1] = np.clip(0.349 * arr[:, :, 0] + 0.686 * arr[:, :, 1] + 0.168 * arr[:, :, 2], 0, 255)
        sepia[:, :, 2] = np.clip(0.272 * arr[:, :, 0] + 0.534 * arr[:, :, 1] + 0.131 * arr[:, :, 2], 0, 255)
        gray = Image.fromarray(sepia.astype(np.uint8))

    buf = io.BytesIO()
    gray.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"color_mode": mode}


def apply_screenshot_conversion(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Frame image inside simulated mobile UI/browser screenshot wrapper."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    # Neutral light UI framing (status bar + margins)
    framed = Image.new("RGB", (w, h), (242, 244, 247))
    inner_w = int(w * 0.86)
    inner_h = int(h * 0.86)
    scaled_inner = img.resize((inner_w, inner_h), resample=Image.Resampling.BILINEAR)
    offset_x = (w - inner_w) // 2
    offset_y = (h - inner_h) // 2
    framed.paste(scaled_inner, (offset_x, offset_y))
    buf = io.BytesIO()
    framed.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"framed_scale": 0.86, "background_color": (242, 244, 247)}


def apply_metadata_removal(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Re-encode image purely from raster pixels, stripping all EXIF, ICC, and comment headers."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Fresh image without any metadata dictionaries
    clean_raster = Image.fromarray(np.array(img, dtype=np.uint8))
    buf = io.BytesIO()
    clean_raster.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), {"stripped_metadata": True}


def apply_multi_pass_reencoding(image_bytes: bytes, passes: int = 3, quality: int = 80) -> Tuple[bytes, Dict[str, Any]]:
    """Simulate multi-platform forwarding by repeatedly re-encoding with JPEG compression."""
    current_bytes = image_bytes
    for p in range(passes):
        img = Image.open(io.BytesIO(current_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        current_bytes = buf.getvalue()
    return current_bytes, {"passes": passes, "quality_per_pass": quality}


def apply_minor_noise(image_bytes: bytes, sigma: float = 3.5, seed: int = 42) -> Tuple[bytes, Dict[str, Any]]:
    """Inject minor additive Gaussian sensor noise."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)
    buf = io.BytesIO()
    noisy_img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {"noise_type": "gaussian", "sigma": sigma, "seed": seed}


TRANSFORMATION_REGISTRY: Dict[str, Callable[[bytes], Tuple[bytes, Dict[str, Any]]]] = {
    "jpeg_compression": lambda b: apply_jpeg_compression(b, quality=50),
    "png_conversion": apply_png_conversion,
    "resizing": lambda b: apply_resizing(b, scale_factor=0.5),
    "cropping": lambda b: apply_cropping(b, crop_ratio=0.8),
    "blur": lambda b: apply_blur(b, radius=2.5),
    "brightness_changes": lambda b: apply_brightness_change(b, factor=1.25),
    "contrast_changes": lambda b: apply_contrast_change(b, factor=0.65),
    "color_transformations": lambda b: apply_color_transformation(b, mode="grayscale"),
    "screenshot_conversion": apply_screenshot_conversion,
    "metadata_removal": apply_metadata_removal,
    "re_encoding": lambda b: apply_multi_pass_reencoding(b, passes=3, quality=80),
    "minor_image_noise": lambda b: apply_minor_noise(b, sigma=3.5),
}


# ---------------------------------------------------------------------------
# Derivative Creator & Provenance Tracker
# ---------------------------------------------------------------------------

def create_robustness_derivative(
    sample_id: str,
    raw_bytes: bytes,
    transformation_name: str,
    is_manipulated: bool,
) -> RobustnessDerivative:
    """Generate a separate test derivative, preserving the pristine original bytes."""
    if transformation_name not in TRANSFORMATION_REGISTRY:
        raise ValueError(f"Unknown transformation: {transformation_name}")

    original_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    transform_fn = TRANSFORMATION_REGISTRY[transformation_name]
    derivative_bytes, params = transform_fn(raw_bytes)

    derivative_sha256 = hashlib.sha256(derivative_bytes).hexdigest()

    # Verification: pristine original must not be mutated
    post_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if post_sha256 != original_sha256:
        raise RuntimeError("CRITICAL ERROR: Original evidence bytes were mutated during derivative creation!")

    return RobustnessDerivative(
        original_sample_id=sample_id,
        transformation_name=transformation_name,
        original_bytes=raw_bytes,
        original_sha256=original_sha256,
        derivative_bytes=derivative_bytes,
        derivative_sha256=derivative_sha256,
        transformation_params=params,
        is_manipulated=is_manipulated,
    )


# ---------------------------------------------------------------------------
# Pair Evaluator & Metric Aggregator
# ---------------------------------------------------------------------------

def evaluate_transformation_pair(
    derivative: RobustnessDerivative,
    detector: BaseImageDetector,
) -> TransformationPairEvaluation:
    """Evaluate detector responses on clean original vs. perturbed derivative."""
    # 1. Clean response
    clean_resp = detector.detect(derivative.original_bytes)

    # 2. Perturbed derivative response
    model_failed = False
    try:
        perturbed_resp = detector.detect(derivative.derivative_bytes)
    except Exception:
        model_failed = True
        perturbed_resp = detector.detect_safe(derivative.derivative_bytes)

    score_delta = float(perturbed_resp.raw_model_score - clean_resp.raw_model_score)
    label_changed = (clean_resp.assessment_label != perturbed_resp.assessment_label)

    # False positive transition: clean was authentic/negative, perturbed became suspicious/manipulated
    is_fp_trans = (
        not derivative.is_manipulated
        and clean_resp.assessment_label == AssessmentLabel.NO_OBVIOUS_ARTIFACTS
        and perturbed_resp.assessment_label in (AssessmentLabel.SUSPICIOUS_MANIPULATION, AssessmentLabel.HIGHLY_LIKELY_MANIPULATED)
    )

    # False negative transition: clean was positive, perturbed became negative or inconclusive
    is_fn_trans = (
        derivative.is_manipulated
        and clean_resp.assessment_label in (AssessmentLabel.SUSPICIOUS_MANIPULATION, AssessmentLabel.HIGHLY_LIKELY_MANIPULATED)
        and perturbed_resp.assessment_label in (AssessmentLabel.NO_OBVIOUS_ARTIFACTS, AssessmentLabel.INCONCLUSIVE)
    )

    # Uncertainty escalation
    escalated = False
    if clean_resp.uncertainty_state in (UncertaintyState.ANALYSIS_USABLE_EVIDENCE, UncertaintyState.LOW):
        if perturbed_resp.uncertainty_state not in (UncertaintyState.ANALYSIS_USABLE_EVIDENCE, UncertaintyState.LOW):
            escalated = True

    return TransformationPairEvaluation(
        sample_id=derivative.original_sample_id,
        transformation_name=derivative.transformation_name,
        is_manipulated=derivative.is_manipulated,
        clean_score=clean_resp.raw_model_score,
        perturbed_score=perturbed_resp.raw_model_score,
        score_delta=round(score_delta, 4),
        clean_label=clean_resp.assessment_label,
        perturbed_label=perturbed_resp.assessment_label,
        label_changed=label_changed,
        is_false_positive_transition=is_fp_trans,
        is_false_negative_transition=is_fn_trans,
        clean_uncertainty=clean_resp.uncertainty_state,
        perturbed_uncertainty=perturbed_resp.uncertainty_state,
        uncertainty_escalated=escalated,
        quality_warnings_triggered=perturbed_resp.quality_warnings or [],
        model_failed=model_failed,
    )


def summarize_transformation_robustness(
    evaluations: List[TransformationPairEvaluation],
) -> TransformationRobustnessSummary:
    """Aggregate individual sample pair evaluations into statistical metrics."""
    if not evaluations:
        raise ValueError("Cannot summarize empty evaluation list.")

    trans_name = evaluations[0].transformation_name
    n = len(evaluations)

    deltas = [e.score_delta for e in evaluations]
    abs_deltas = [abs(d) for d in deltas]

    mean_drift = float(np.mean(deltas))
    max_drift = float(np.max(abs_deltas))
    std_drift = float(np.std(deltas))

    label_flips = sum(1 for e in evaluations if e.label_changed)
    fp_transitions = sum(1 for e in evaluations if e.is_false_positive_transition)
    fn_transitions = sum(1 for e in evaluations if e.is_false_negative_transition)
    uncertainty_escalations = sum(1 for e in evaluations if e.uncertainty_escalated)
    model_failures = sum(1 for e in evaluations if e.model_failed)
    warned_samples = sum(1 for e in evaluations if len(e.quality_warnings_triggered) > 0)

    # Observations synthesis
    obs: List[str] = []
    if model_failures > 0:
        obs.append(f"CRITICAL: {model_failures} runtime model crashes encountered.")
    if fp_transitions > 0:
        obs.append(f"HAZARD: {fp_transitions} authentic images falsely flipped to manipulated.")
    if fn_transitions > 0:
        obs.append(f"EVASION: {fn_transitions} manipulated images masked into negative triage.")
    if uncertainty_escalations > 0:
        obs.append(f"Uncertainty escalated in {uncertainty_escalations}/{n} samples ({uncertainty_escalations/n*100:.1f}%).")
    if warned_samples > 0:
        obs.append(f"Quality engine intercepted transformation in {warned_samples}/{n} samples ({warned_samples/n*100:.1f}%).")
    if not obs:
        obs.append("Stable: Minimal score drift with zero label flipping or false transitions.")

    return TransformationRobustnessSummary(
        transformation_name=trans_name,
        sample_count=n,
        mean_score_drift=round(mean_drift, 4),
        max_score_drift=round(max_drift, 4),
        score_std_dev=round(std_drift, 4),
        label_flip_rate=round(label_flips / n, 4),
        false_positive_transitions=fp_transitions,
        false_negative_transitions=fn_transitions,
        uncertainty_escalation_rate=round(uncertainty_escalations / n, 4),
        model_failure_count=model_failures,
        quality_warning_trigger_rate=round(warned_samples / n, 4),
        observations=" ".join(obs),
    )
