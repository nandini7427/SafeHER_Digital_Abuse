"""SafeHER Image Quality and Uncertainty Assessment Engine.

Evaluates:
- Image resolution and aspect ratio (cropping).
- Excessive blur via Laplacian variance.
- Compression artifacts via 8x8 DCT grid blockiness.
- Extreme brightness (overexposure highlight clipping) and darkness (underexposure shadow clipping).
- Monochromatic or flat synthetic graphics.
- Insufficient face resolution and area ratio.
- Corrupted or truncated image data.

Distinguishes 6 discrete uncertainty and forensic usability states:
1. ANALYSIS_USABLE_EVIDENCE
2. ANALYSIS_POOR_QUALITY_EVIDENCE
3. MODEL_UNCERTAIN
4. MODEL_UNAVAILABLE
5. OUTSIDE_SUPPORTED_DOMAIN
6. ANALYSIS_CANNOT_BE_COMPLETED
"""

from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ImageQualityIndicators,
    UncertaintyState,
)
from ai_core.image_intelligence.ood_detector import (
    DomainCategory,
    OutOfDistributionDetector,
)

# ---------------------------------------------------------------------------
# Scientifically Defensible Quality Thresholds
# ---------------------------------------------------------------------------

# Resolution thresholds
MIN_FORENSIC_DIMENSION: int = 128
OPTIMAL_FORENSIC_DIMENSION: int = 256
MIN_ASPECT_RATIO: float = 0.20  # 1:5 ratio (severely cropped ribbon)
MAX_ASPECT_RATIO: float = 5.00  # 5:1 ratio (panoramic banner)

# Blur thresholds (discrete Laplacian variance on normalized grayscale)
BLUR_THRESHOLD_LAPLACIAN_VAR: float = 15.0
MODERATE_BLUR_THRESHOLD: float = 35.0

# Compression blockiness thresholds (8x8 DCT boundary step vs interior step)
COMPRESSION_BLOCKINESS_THRESHOLD: float = 2.5

# Exposure thresholds (normalized 0-255 luminance)
OVEREXPOSURE_RATIO_THRESHOLD: float = 0.30  # > 30% of pixels >= 250
OVEREXPOSURE_MEAN_THRESHOLD: float = 240.0   # mean luminance > 240
UNDEREXPOSURE_RATIO_THRESHOLD: float = 0.35  # > 35% of pixels <= 10
UNDEREXPOSURE_MEAN_THRESHOLD: float = 15.0   # mean luminance < 15

# Monochromatic / flat surface threshold (variance across all RGB channels)
FLAT_SURFACE_VARIANCE_THRESHOLD: float = 5.0

# Face sufficiency thresholds (when face detection is evaluated)
MIN_FACE_DIMENSION: int = 60       # minimum 60x60 px
MIN_FACE_AREA_RATIO: float = 0.02  # at least 2% of image area


# ---------------------------------------------------------------------------
# Algorithmic Metric Functions
# ---------------------------------------------------------------------------

def compute_laplacian_variance(gray_arr: np.ndarray) -> float:
    """Compute the variance of the 2D discrete Laplacian filter on grayscale pixels.

    Uses standard 3x3 Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]].
    Sharp, in-focus photographic images exhibit variance > 50-500.
    Heavily blurred images (e.g. Gaussian sigma >= 4.0) collapse below 15.0.
    """
    if gray_arr.shape[0] < 3 or gray_arr.shape[1] < 3:
        return 0.0

    # Vectorized 2D interior Laplacian convolution
    lap = (
        gray_arr[:-2, 1:-1]
        + gray_arr[2:, 1:-1]
        + gray_arr[1:-1, :-2]
        + gray_arr[1:-1, 2:]
        - 4.0 * gray_arr[1:-1, 1:-1]
    )
    return float(np.var(lap))


def compute_compression_blockiness(gray_arr: np.ndarray) -> float:
    """Measure the ratio of 8x8 DCT grid boundary discontinuities vs intra-block differences.

    In lossy JPEG compression with low quality (Q < 50), quantization steps at
    multiples of 8 pixels produce sharp inter-block steps relative to intra-block pixels.
    """
    h, w = gray_arr.shape
    if h < 24 or w < 24:
        return 1.0

    # Vertical boundaries: differences between column c and c+1
    diffs_x = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])

    # Indices of columns at 8-pixel boundaries: 7, 15, 23, ...
    boundary_cols = [c for c in range(7, w - 1, 8)]
    interior_cols = [c for c in range(w - 1) if c not in boundary_cols]

    if not boundary_cols or not interior_cols:
        return 1.0

    boundary_mean_x = float(np.mean(diffs_x[:, boundary_cols]))
    interior_mean_x = float(np.mean(diffs_x[:, interior_cols]))

    # Horizontal boundaries: differences between row r and r+1
    diffs_y = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])
    boundary_rows = [r for r in range(7, h - 1, 8)]
    interior_rows = [r for r in range(h - 1) if r not in boundary_rows]

    if not boundary_rows or not interior_rows:
        return 1.0

    boundary_mean_y = float(np.mean(diffs_y[boundary_rows, :]))
    interior_mean_y = float(np.mean(diffs_y[interior_rows, :]))

    boundary_mean = (boundary_mean_x + boundary_mean_y) / 2.0
    interior_mean = (interior_mean_x + interior_mean_y) / 2.0

    if interior_mean < 1e-3:
        return 1.0

    return float(boundary_mean / interior_mean)


def compute_exposure_metrics(gray_arr: np.ndarray) -> Tuple[float, float, float, str]:
    """Compute mean luminance, overexposed ratio, underexposed ratio, and exposure level."""
    mean_lum = float(np.mean(gray_arr))
    over_ratio = float(np.mean(gray_arr >= 250.0))
    under_ratio = float(np.mean(gray_arr <= 10.0))

    if over_ratio >= OVEREXPOSURE_RATIO_THRESHOLD or mean_lum >= OVEREXPOSURE_MEAN_THRESHOLD:
        level = "overexposed"
    elif under_ratio >= UNDEREXPOSURE_RATIO_THRESHOLD or mean_lum <= UNDEREXPOSURE_MEAN_THRESHOLD:
        level = "underexposed"
    elif over_ratio >= 0.15 and under_ratio >= 0.15:
        level = "extreme_contrast"
    else:
        level = "normal"

    return mean_lum, over_ratio, under_ratio, level


def check_monochromatic_or_flat(rgb_arr: np.ndarray) -> bool:
    """Return True if the image is a solid color, near-zero variance flat graphic, or synthetic block."""
    total_var = float(np.var(rgb_arr))
    return total_var < FLAT_SURFACE_VARIANCE_THRESHOLD


def check_screen_capture(width: int, height: int, is_png: bool = False, has_exif: bool = False) -> bool:
    """Heuristic identification of mobile or desktop screen captures."""
    # Common smartphone/display screen capture resolutions and aspect ratios
    aspect = max(width, height) / max(1, min(width, height))
    is_tall_phone_ratio = 1.95 <= aspect <= 2.25  # Modern 19.5:9 to 20:9 displays
    common_screen_dims = {
        (1080, 1920), (1920, 1080),
        (1170, 2532), (2532, 1170),
        (1284, 2778), (2778, 1284),
        (1179, 2556), (2556, 1179),
        (1290, 2796), (2796, 1290),
        (1440, 2560), (2560, 1440),
        (1440, 3120), (3120, 1440),
    }
    dim_match = (width, height) in common_screen_dims
    # PNG format without camera EXIF and tall aspect ratio strongly indicates screen grab
    if is_png and not has_exif and is_tall_phone_ratio:
        return True
    return dim_match and not has_exif


def evaluate_face_sufficiency(
    img_width: int,
    img_height: int,
    face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Tuple[Optional[bool], int, Optional[bool], Optional[float]]:
    """Evaluate face sufficiency metrics given normalized (ymin, xmin, ymax, xmax) boxes."""
    if face_bboxes is None:
        return None, 0, None, None

    face_count = len(face_bboxes)
    if face_count == 0:
        return False, 0, False, 0.0

    total_area = float(img_width * img_height)
    max_area_ratio = 0.0
    all_faces_ok = True

    for (ymin, xmin, ymax, xmax) in face_bboxes:
        fw_px = (xmax - xmin) * img_width
        fh_px = (ymax - ymin) * img_height
        face_area = fw_px * fh_px
        area_ratio = face_area / total_area if total_area > 0 else 0.0

        if area_ratio > max_area_ratio:
            max_area_ratio = area_ratio

        if fw_px < MIN_FACE_DIMENSION or fh_px < MIN_FACE_DIMENSION or area_ratio < MIN_FACE_AREA_RATIO:
            all_faces_ok = False

    return True, face_count, all_faces_ok, float(max_area_ratio)


# ---------------------------------------------------------------------------
# Comprehensive Evaluation Container & Pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityAssessmentResult:
    """Full forensic evaluation result including indicators, warnings, and recommended actions."""

    indicators: ImageQualityIndicators
    warnings: List[str]
    uncertainty_state: UncertaintyState
    analysis_eligibility: bool
    reason_for_insufficient_evidence: Optional[str]
    recommended_next_step: str
    assessment_label: AssessmentLabel


def evaluate_image_quality(
    img: Image.Image,
    raw_score: Optional[float] = None,
    face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    model_failed: bool = False,
    is_corrupted: bool = False,
    format_name: Optional[str] = None,
    has_exif: bool = False,
) -> QualityAssessmentResult:
    """Perform complete forensic image quality and uncertainty triage.

    Args:
        img: Decoded PIL Image (working copy).
        raw_score: Raw model inference score if available.
        face_bboxes: Optional list of normalized face bounding boxes.
        model_failed: True if the primary detection runner raised an exception or was unavailable.
        is_corrupted: True if byte stream had corruption artifacts.
        format_name: Detected format string (e.g. 'JPEG', 'PNG').
        has_exif: True if camera EXIF metadata was present.

    Returns:
        QualityAssessmentResult with all indicators, uncertainty state, and recommendations.
    """
    width, height = img.size
    aspect_ratio = float(width / max(1, height))
    megapixels = float(width * height / 1_000_000.0)

    # 1. Corrupted Data Check
    if is_corrupted:
        indicators = ImageQualityIndicators(
            width=width,
            height=height,
            megapixels=megapixels,
            aspect_ratio=aspect_ratio,
            resolution_ok=False,
            blur_score=0.0,
            is_blurry=True,
            compression_artifact_score=1.0,
            has_heavy_compression=False,
            mean_luminance=0.0,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_level="normal",
            is_monochromatic_or_flat=False,
            is_screen_capture=False,
            face_detected=None,
            face_count=0,
            face_resolution_ok=None,
            face_size_ratio=None,
            corrupted_data_detected=True,
            domain_category=DomainCategory.UNSUPPORTED_FORMAT.value,
            is_in_domain=False,
            overall_quality_score=0.0,
        )
        return QualityAssessmentResult(
            indicators=indicators,
            warnings=["Image payload data contains corrupted or truncated stream bytes."],
            uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED,
            analysis_eligibility=False,
            reason_for_insufficient_evidence="Corrupted or truncated image data prevents reliable forensic decoding.",
            recommended_next_step="Re-export or re-download the original file to ensure it is not truncated or corrupted.",
            assessment_label=AssessmentLabel.INCONCLUSIVE,
        )

    # Convert to numpy arrays for vectorized processing
    rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
    # ITU-R BT.601 luminance
    gray_arr = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]

    # 2. Individual Quality Computations
    resolution_ok = (width >= MIN_FORENSIC_DIMENSION and height >= MIN_FORENSIC_DIMENSION)
    blur_score = compute_laplacian_variance(gray_arr)
    is_blurry = (blur_score < BLUR_THRESHOLD_LAPLACIAN_VAR)

    compression_artifact_score = compute_compression_blockiness(gray_arr)
    has_heavy_compression = (compression_artifact_score > COMPRESSION_BLOCKINESS_THRESHOLD)

    mean_lum, over_ratio, under_ratio, exposure_level = compute_exposure_metrics(gray_arr)
    is_flat = check_monochromatic_or_flat(rgb_arr)
    is_screen = check_screen_capture(width, height, is_png=(format_name == "PNG"), has_exif=has_exif)

    face_detected, face_count, face_res_ok, face_ratio = evaluate_face_sufficiency(
        width, height, face_bboxes=face_bboxes
    )

    # 3. Domain & Out-of-Distribution (OOD) Classification
    ood_detector = OutOfDistributionDetector()
    domain_res = ood_detector.classify_domain(
        img=img,
        format_name=format_name,
        has_exif=has_exif,
        face_bboxes=face_bboxes,
        is_corrupted=is_corrupted,
        compression_blockiness=compression_artifact_score,
    )

    # 3. Assemble Quality Warnings
    warnings: List[str] = []

    if not resolution_ok:
        warnings.append(
            f"Image dimensions ({width}x{height} px) are below the minimum forensic requirement of "
            f"{MIN_FORENSIC_DIMENSION}x{MIN_FORENSIC_DIMENSION} px; high-frequency compression artifacts cannot be resolved."
        )

    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        warnings.append(
            f"Extreme aspect ratio ({aspect_ratio:.2f}) indicates severe cropping, panoramic ribbon, or banner format."
        )

    if is_blurry:
        warnings.append(
            f"Excessive blur detected (Laplacian edge variance: {blur_score:.1f} < {BLUR_THRESHOLD_LAPLACIAN_VAR}); "
            "optical or motion defocus destroys fine forensic residuals."
        )

    if has_heavy_compression:
        warnings.append(
            f"Severe compression artifacts detected (8x8 DCT grid blockiness ratio: {compression_artifact_score:.2f} > "
            f"{COMPRESSION_BLOCKINESS_THRESHOLD}); platform recompression obscures manipulation boundaries."
        )

    if exposure_level == "overexposed":
        warnings.append(
            f"Extreme brightness / blown-out highlight clipping detected ({over_ratio * 100:.1f}% saturated pixels, "
            f"mean luminance: {mean_lum:.1f}); pixel saturation destroys forensic data."
        )
    elif exposure_level == "underexposed":
        warnings.append(
            f"Extreme darkness / crushed shadow clipping detected ({under_ratio * 100:.1f}% dark pixels, "
            f"mean luminance: {mean_lum:.1f}); sensor noise floor obscures image content."
        )
    elif exposure_level == "extreme_contrast":
        warnings.append("Extreme dynamic range contrast clipping detected across highlights and shadows.")

    if is_flat:
        warnings.append(
            "Monochromatic or near-zero variance flat graphic detected; lacks natural photographic sensor texture."
        )

    if is_screen:
        warnings.append(
            "Screen capture characteristics detected; UI compositing and system re-rasterization interfere with camera sensor forensics."
        )

    if face_res_ok is False:
        warnings.append(
            f"Detected face regions fail minimum micro-forensic size (under {MIN_FACE_DIMENSION}x{MIN_FACE_DIMENSION} px "
            f"or < {MIN_FACE_AREA_RATIO * 100:.0f}% frame area); insufficient for facial manipulation triage."
        )

    # 4. Composite Quality Score (0.0 to 1.0)
    quality_score = 1.0
    if not resolution_ok:
        quality_score -= 0.35
    elif width < OPTIMAL_FORENSIC_DIMENSION or height < OPTIMAL_FORENSIC_DIMENSION:
        quality_score -= 0.15

    if is_blurry:
        quality_score -= 0.30
    if has_heavy_compression:
        quality_score -= 0.20
    if exposure_level in ("overexposed", "underexposed"):
        quality_score -= 0.25
    if is_flat:
        quality_score -= 0.40
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        quality_score -= 0.20
    if face_res_ok is False:
        quality_score -= 0.20

    overall_quality_score = float(max(0.0, min(1.0, quality_score)))

    indicators = ImageQualityIndicators(
        width=width,
        height=height,
        megapixels=megapixels,
        aspect_ratio=aspect_ratio,
        resolution_ok=resolution_ok,
        blur_score=blur_score,
        is_blurry=is_blurry,
        compression_artifact_score=compression_artifact_score,
        has_heavy_compression=has_heavy_compression,
        mean_luminance=mean_lum,
        overexposed_ratio=over_ratio,
        underexposed_ratio=under_ratio,
        exposure_level=exposure_level,
        is_monochromatic_or_flat=is_flat,
        is_screen_capture=is_screen,
        face_detected=face_detected,
        face_count=face_count,
        face_resolution_ok=face_res_ok,
        face_size_ratio=face_ratio,
        corrupted_data_detected=False,
        domain_category=domain_res.category.value,
        is_in_domain=domain_res.is_in_domain,
        overall_quality_score=overall_quality_score,
    )

    # 5. Determine State from the 6 Discrete States
    # State 5 (Part 1): OUTSIDE_SUPPORTED_DOMAIN for geometric or flat graphical mismatches
    if is_flat or aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return QualityAssessmentResult(
            indicators=indicators,
            warnings=warnings,
            uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
            analysis_eligibility=False,
            reason_for_insufficient_evidence=(
                "Image characteristics (e.g. flat non-photographic graphics or extreme cropping) "
                "fall outside the supported photographic forensic domain."
            ),
            recommended_next_step=(
                "Submit standard photographic evidence containing camera sensor details rather than "
                "digital graphics, solid test squares, or panoramic strips."
            ),
            assessment_label=AssessmentLabel.INCONCLUSIVE,
        )

    # State 2: ANALYSIS_POOR_QUALITY_EVIDENCE
    has_severe_quality_defect = (
        not resolution_ok
        or is_blurry
        or has_heavy_compression
        or exposure_level in ("overexposed", "underexposed")
        or (face_res_ok is False and face_detected is True)
    )

    if has_severe_quality_defect:
        reasons: List[str] = []
        if not resolution_ok:
            reasons.append("resolution is too low (< 128px)")
        if is_blurry:
            reasons.append("image is excessively blurry")
        if has_heavy_compression:
            reasons.append("heavy compression blocking artifacts")
        if exposure_level in ("overexposed", "underexposed"):
            reasons.append(f"extreme {exposure_level} clipping")
        if face_res_ok is False and face_detected is True:
            reasons.append("face resolution is insufficient for micro-forensic analysis")

        return QualityAssessmentResult(
            indicators=indicators,
            warnings=warnings,
            uncertainty_state=UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
            analysis_eligibility=False,
            reason_for_insufficient_evidence=f"Evidence quality is insufficient for reliable triage: {', '.join(reasons)}.",
            recommended_next_step=(
                "Request the original full-resolution camera file directly from the source device. "
                "Avoid images forwarded through social messaging apps or screenshots."
            ),
            # CRITICAL MANDATE: Never classify poor quality images as authentic or fake!
            assessment_label=AssessmentLabel.INCONCLUSIVE,
        )

    # State 5 (Part 2): OUTSIDE_SUPPORTED_DOMAIN from domain classifier
    if not domain_res.is_in_domain and domain_res.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN:
        all_domain_warnings = list(dict.fromkeys(warnings + domain_res.warnings))
        reason = domain_res.reason or (
            f"Image belongs to an unsupported out-of-distribution category ({domain_res.category.value}); "
            "outside validated photographic deepfake domain."
        )
        rec_step = domain_res.recommended_action

        return QualityAssessmentResult(
            indicators=indicators,
            warnings=all_domain_warnings,
            uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
            analysis_eligibility=False,
            reason_for_insufficient_evidence=reason,
            recommended_next_step=rec_step,
            assessment_label=AssessmentLabel.INCONCLUSIVE,
        )

    # State 4: MODEL_UNAVAILABLE
    if model_failed:
        return QualityAssessmentResult(
            indicators=indicators,
            warnings=warnings + ["Primary forensic detection runner was unavailable or encountered a runtime error."],
            uncertainty_state=UncertaintyState.MODEL_UNAVAILABLE,
            analysis_eligibility=False,
            reason_for_insufficient_evidence="The primary forensic detection model is currently offline or unavailable.",
            recommended_next_step="Verify backend model service status and retry the assessment.",
            assessment_label=AssessmentLabel.INCONCLUSIVE,
        )

    # State 3: MODEL_UNCERTAIN
    # Usable evidence, but model score falls in borderline ambiguous zone [0.40, 0.60]
    score = raw_score if raw_score is not None else 0.10
    if 0.40 <= score <= 0.60:
        return QualityAssessmentResult(
            indicators=indicators,
            warnings=warnings + ["Detection anomaly score falls directly near the decision boundary."],
            uncertainty_state=UncertaintyState.MODEL_UNCERTAIN,
            analysis_eligibility=True,
            reason_for_insufficient_evidence=(
                "Forensic model outputs intermediate anomaly signals near the decision threshold; "
                "signals are inconclusive."
            ),
            recommended_next_step=(
                "Submit additional corroborating images or forward this evidence to an accredited digital forensic examiner."
            ),
            assessment_label=AssessmentLabel.SUSPICIOUS_MANIPULATION if score >= 0.45 else AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
        )

    # State 1: ANALYSIS_USABLE_EVIDENCE
    # Usable quality, unambiguous score (< 0.40 or > 0.60)
    label = (
        AssessmentLabel.HIGHLY_LIKELY_MANIPULATED
        if score >= 0.70
        else (AssessmentLabel.SUSPICIOUS_MANIPULATION if score >= 0.45 else AssessmentLabel.NO_OBVIOUS_ARTIFACTS)
    )

    return QualityAssessmentResult(
        indicators=indicators,
        warnings=warnings,
        uncertainty_state=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
        analysis_eligibility=True,
        reason_for_insufficient_evidence=None,
        recommended_next_step=(
            "Evidence quality is usable for automated triage. Maintain original file cryptographic hash (SHA-256) "
            "and preserve chain of custody."
        ),
        assessment_label=label,
    )
