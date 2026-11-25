"""SafeHER Out-of-Distribution (OOD) Benchmark and Domain Suitability Evaluation Suite.

Generates reproducible controlled evaluation samples across in-domain and out-of-distribution
categories to evaluate SafeHER's domain recognition, false rejection, false acceptance,
and uncertainty behavior.

Categories:
1. In-Domain:
   - Photographic Portraits (natural camera portraits with human subjects)
   - Photographic Scenes (continuous tones, optical blur, sensor PRNU)
2. Out-of-Distribution (OOD):
   - Drawings & Sketches (pencil/charcoal line art on paper)
   - Digital Illustrations (flat vector graphics, cartoons, low color entropy)
   - Memes & Text Graphics (caption banners, rasterized text overlays)
   - Screenshots & Mobile UI (tall aspect ratio, status bars, chat bubbles, no EXIF)
   - Medical Imagery (monochromatic radiograms with perimeter masking)
   - Documents & Scanned Text (high white background, dark text lines, low mid-tones)
   - Landscapes without Subject (outdoor scenery tested under facial triage)
   - Extremely Compressed Images (lossy 8x8 DCT grid blocking)
   - Unsupported / Corrupted Formats (corrupted byte streams)
"""

from dataclasses import dataclass
import hashlib
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.ood_detector import (
    DomainCategory,
    OutOfDistributionDetector,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    UncertaintyState,
)


@dataclass(frozen=True)
class OODSample:
    """Benchmark evaluation sample with ground-truth domain category."""

    sample_id: str
    category: DomainCategory
    is_in_domain: bool
    data: bytes
    sha256: str
    metadata: Dict[str, Any]


def generate_photographic_portrait(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate synthetic photographic portrait with continuous skin tones and sensor noise."""
    rng = np.random.RandomState(seed)
    w, h = size
    # Warm photographic skin-tone background with natural gradient
    y_grad = np.linspace(190, 220, h)[:, None]
    x_grad = np.linspace(150, 180, w)[None, :]
    r = np.clip(0.6 * y_grad + 0.4 * x_grad, 0, 255)
    g = np.clip(r * 0.78, 0, 255)
    b = np.clip(r * 0.65, 0, 255)
    arr = np.stack([r, g, b], axis=-1)

    # Face ellipse in the center
    yy, xx = np.mgrid[:h, :w]
    face_mask = (((xx - w / 2) / (w * 0.28)) ** 2 + ((yy - h / 2) / (h * 0.35)) ** 2) <= 1.0
    arr[face_mask, 0] = np.clip(arr[face_mask, 0] * 1.15, 0, 255)
    arr[face_mask, 1] = np.clip(arr[face_mask, 1] * 1.05, 0, 255)
    arr[face_mask, 2] = np.clip(arr[face_mask, 2] * 0.95, 0, 255)

    # Sensor noise
    sensor_noise = rng.normal(0, 3.5, arr.shape)
    arr = np.clip(arr + sensor_noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def generate_photographic_scene(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate synthetic photographic scenery with rich color entropy and natural gradients."""
    rng = np.random.RandomState(seed)
    w, h = size
    sky_h = int(h * 0.5)
    ground_h = h - sky_h

    # Sky: cyan/blue gradient
    sky_r = np.repeat(np.linspace(100, 180, sky_h)[:, None], w, axis=1)
    sky_g = np.repeat(np.linspace(150, 210, sky_h)[:, None], w, axis=1)
    sky_b = np.repeat(np.linspace(220, 250, sky_h)[:, None], w, axis=1)

    # Ground: green/brown gradient
    ground_r = np.repeat(np.linspace(80, 130, ground_h)[:, None], w, axis=1)
    ground_g = np.repeat(np.linspace(140, 100, ground_h)[:, None], w, axis=1)
    ground_b = np.repeat(np.linspace(40, 70, ground_h)[:, None], w, axis=1)

    r = np.vstack([sky_r, ground_r])
    g = np.vstack([sky_g, ground_g])
    b = np.vstack([sky_b, ground_b])

    arr = np.stack([r, g, b], axis=-1)
    # Sensor noise
    arr = np.clip(arr + rng.normal(0, 4.0, arr.shape), 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_drawing_or_sketch(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate pencil line sketch on textured off-white paper."""
    rng = np.random.RandomState(seed)
    w, h = size
    # Light paper background (mean ~235)
    arr = np.full((h, w, 3), 238, dtype=np.float32)
    arr += rng.normal(0, 2.0, arr.shape)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw dark pencil contours
    for _ in range(8):
        x0, y0 = rng.randint(20, w - 20), rng.randint(20, h - 20)
        x1, y1 = rng.randint(20, w - 20), rng.randint(20, h - 20)
        draw.line([(x0, y0), (x1, y1)], fill=(45, 45, 45), width=2)

    # Pencil circle
    draw.ellipse([w * 0.25, h * 0.25, w * 0.75, h * 0.75], outline=(40, 40, 40), width=2)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_digital_illustration(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate flat cartoon / vector illustration with saturated shapes and low entropy."""
    w, h = size
    # Saturated flat cyan background
    img = Image.new("RGB", (w, h), color=(40, 180, 230))
    draw = ImageDraw.Draw(img)

    # Large saturated yellow circle
    draw.ellipse([w * 0.15, h * 0.15, w * 0.85, h * 0.85], fill=(255, 215, 0), outline=None)
    # Magenta rectangle
    draw.rectangle([w * 0.3, h * 0.4, w * 0.7, h * 0.8], fill=(220, 20, 100), outline=None)
    # White polygon
    draw.polygon([(w * 0.5, h * 0.2), (w * 0.4, h * 0.35), (w * 0.6, h * 0.35)], fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_meme_or_text_graphic(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate image with synthetic top/bottom caption text banner."""
    base_bytes = generate_photographic_scene(seed=seed, size=size)
    img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = size

    # Solid white top banner (15% height)
    banner_h = int(h * 0.15)
    draw.rectangle([0, 0, w, banner_h], fill=(255, 255, 255))
    # Simulated bold black impact text lines in banner
    for r in range(4, banner_h - 4, 6):
        draw.line([10, r, w - 10, r], fill=(0, 0, 0), width=3)

    # Bottom banner (15% height)
    draw.rectangle([0, h - banner_h, w, h], fill=(0, 0, 0))
    for r in range(h - banner_h + 4, h - 4, 6):
        draw.line([10, r, w - 10, r], fill=(255, 255, 255), width=3)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_screenshot_or_ui(seed: int = 42, size: Tuple[int, int] = (270, 585)) -> bytes:
    """Generate mobile device screenshot with top status bar and flat UI bubbles."""
    w, h = size
    # Mobile app background
    img = Image.new("RGB", (w, h), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)

    # Top status bar (solid uniform dark bar)
    bar_h = int(h * 0.035)
    draw.rectangle([0, 0, w, bar_h], fill=(20, 20, 20))

    # Chat bubbles
    draw.rectangle([20, 80, w - 60, 140], fill=(255, 255, 255), outline=(220, 220, 220))
    draw.rectangle([60, 160, w - 20, 220], fill=(0, 132, 255))

    buf = io.BytesIO()
    # PNG format without camera EXIF
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_medical_imagery(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate simulated chest X-ray radiogram with perimeter black border and zero chrominance."""
    rng = np.random.RandomState(seed)
    w, h = size
    # Monochromatic grayscale
    gray = np.zeros((h, w), dtype=np.float32)

    # Interior rib/lung structures
    inner_mask = np.zeros((h, w), dtype=bool)
    pad = int(min(w, h) * 0.08)
    inner_mask[pad:-pad, pad:-pad] = True

    # Soft tissue luminance in interior (30 to 190)
    yy, xx = np.mgrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    rad = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    gray[inner_mask] = np.clip(180.0 - 0.7 * rad[inner_mask] + rng.normal(0, 10.0, np.sum(inner_mask)), 25, 210)

    # Perimeter 5% border is pitch black (0.0)
    gray[:pad, :] = 0.0
    gray[-pad:, :] = 0.0
    gray[:, :pad] = 0.0
    gray[:, -pad:] = 0.0

    rgb_arr = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb_arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_document_or_scanned_text(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate scanned text document with white paper and dark horizontal text lines."""
    w, h = size
    # White paper background
    img = Image.new("RGB", (w, h), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)

    # 15 horizontal typed text lines
    for line_y in range(30, h - 30, 14):
        # Text line segments with word spaces
        draw.line([30, line_y, w - 30, line_y], fill=(30, 30, 30), width=2)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_photographic_portrait_medium(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate medium-shot photographic portrait with human subject, clothing, and natural background."""
    rng = np.random.RandomState(seed)
    w, h = size
    # Neutral photographic background
    arr = np.full((h, w, 3), 165.0, dtype=np.float32)
    # Head and shoulders human subject
    yy, xx = np.mgrid[:h, :w]
    head_mask = (((xx - w / 2) / (w * 0.20)) ** 2 + ((yy - h * 0.35) / (h * 0.22)) ** 2) <= 1.0
    torso_mask = (yy >= h * 0.55) & (np.abs(xx - w / 2) <= w * 0.38)

    # Skin tone on head
    arr[head_mask, 0] = 210.0
    arr[head_mask, 1] = 165.0
    arr[head_mask, 2] = 135.0

    # Clothing on torso
    arr[torso_mask, 0] = 50.0
    arr[torso_mask, 1] = 70.0
    arr[torso_mask, 2] = 110.0

    # Camera sensor noise
    arr = np.clip(arr + rng.normal(0, 3.0, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def generate_landscape_without_subject(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate outdoor landscape tested for face-only triage tasks."""
    return generate_photographic_scene(seed=seed, size=size)


def generate_extremely_compressed(seed: int = 42, size: Tuple[int, int] = (256, 256)) -> bytes:
    """Generate image with heavy lossy JPEG compression destroying sensor PRNU."""
    base_bytes = generate_photographic_scene(seed=seed, size=size)
    img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
    buf = io.BytesIO()
    # Heavy compression at quality=5
    img.save(buf, format="JPEG", quality=5)
    return buf.getvalue()


def generate_unsupported_format(seed: int = 42) -> bytes:
    """Generate corrupted byte stream."""
    return b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01" + b"\x00" * 20


def build_ood_benchmark_dataset(samples_per_category: int = 5) -> List[OODSample]:
    """Assemble a balanced benchmark dataset spanning all 11 supported and unsupported categories."""
    dataset: List[OODSample] = []

    generators = [
        # In-Domain
        ("photo_portrait", DomainCategory.PHOTOGRAPHIC_PORTRAIT, True, generate_photographic_portrait),
        ("photo_portrait_medium", DomainCategory.PHOTOGRAPHIC_PORTRAIT, True, generate_photographic_portrait_medium),
        # Out-of-Distribution
        ("drawing", DomainCategory.DRAWING_OR_SKETCH, False, generate_drawing_or_sketch),
        ("illustration", DomainCategory.DIGITAL_ILLUSTRATION, False, generate_digital_illustration),
        ("meme", DomainCategory.MEME_OR_TEXT_GRAPHIC, False, generate_meme_or_text_graphic),
        ("screenshot", DomainCategory.SCREENSHOT_OR_UI, False, generate_screenshot_or_ui),
        ("medical", DomainCategory.MEDICAL_IMAGERY, False, generate_medical_imagery),
        ("document", DomainCategory.DOCUMENT_OR_SCANNED_TEXT, False, generate_document_or_scanned_text),
        ("landscape_noface", DomainCategory.LANDSCAPE_WITHOUT_SUBJECT, False, generate_landscape_without_subject),
        ("extreme_compression", DomainCategory.EXTREMELY_COMPRESSED, False, generate_extremely_compressed),
        ("unsupported_format", DomainCategory.UNSUPPORTED_FORMAT, False, generate_unsupported_format),
    ]

    for prefix, cat, in_domain, gen_fn in generators:
        for idx in range(samples_per_category):
            seed = 100 + idx * 7
            try:
                data = gen_fn(seed=seed)
            except TypeError:
                data = gen_fn()

            sha256 = hashlib.sha256(data).hexdigest()
            dataset.append(
                OODSample(
                    sample_id=f"{prefix}_{idx:02d}",
                    category=cat,
                    is_in_domain=in_domain,
                    data=data,
                    sha256=sha256,
                    metadata={"seed": seed, "prefix": prefix},
                )
            )

    return dataset


@dataclass(frozen=True)
class OODEvaluationSummary:
    """Comprehensive performance metrics for Out-of-Distribution detection."""

    total_samples: int
    in_domain_samples: int
    ood_samples: int
    true_positives: int  # Correctly identified as in-domain
    false_positives: int  # OOD incorrectly accepted as in-domain (FAR)
    true_negatives: int  # Correctly identified as OOD
    false_negatives: int  # In-domain incorrectly rejected as OOD (FRR)
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_acceptance_rate: float
    false_rejection_rate: float
    forced_binary_on_ood_count: int  # MUST BE ZERO
    inconclusive_on_ood_rate: float  # MUST BE 100%
    per_category_metrics: Dict[str, Dict[str, Any]]
    sha256_invariance_verified: bool


def evaluate_ood_detection_benchmark(
    dataset: List[OODSample],
    detector: Optional[SafeHERImageDetector] = None,
) -> OODEvaluationSummary:
    """Execute end-to-end evaluation on the benchmark dataset."""
    det = detector or SafeHERImageDetector()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    forced_binary_on_ood = 0
    inconclusive_on_ood = 0
    all_sha256_invariant = True

    per_category: Dict[str, Dict[str, Any]] = {}

    for sample in dataset:
        # 1. Cryptographic pre-hash check
        initial_hash = hashlib.sha256(sample.data).hexdigest()

        # 2. Select detection task (e.g. FACE_MANIPULATION for landscape_noface)
        task = DetectionTask.FACE_MANIPULATION if sample.category == DomainCategory.LANDSCAPE_WITHOUT_SUBJECT else DetectionTask.HYBRID_MANIPULATION

        # 3. Safe detection execution
        response = det.detect_safe(data=sample.data)

        # 4. Cryptographic post-hash verification (zero mutation)
        post_hash = hashlib.sha256(sample.data).hexdigest()
        if initial_hash != post_hash or initial_hash != sample.sha256:
            all_sha256_invariant = False

        # In-domain decision: analysis_eligibility is True and not OUTSIDE_SUPPORTED_DOMAIN
        predicted_in_domain = response.analysis_eligibility and response.uncertainty_state != UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN

        cat_name = sample.category.value
        if cat_name not in per_category:
            per_category[cat_name] = {
                "count": 0,
                "correct_domain": 0,
                "predicted_categories": {},
                "uncertainty_states": {},
                "assessment_labels": {},
                "inconclusive_count": 0,
            }

        cat_stats = per_category[cat_name]
        cat_stats["count"] += 1

        # Uncertainty & Label recording
        u_state = response.uncertainty_state.value
        cat_stats["uncertainty_states"][u_state] = cat_stats["uncertainty_states"].get(u_state, 0) + 1
        lbl = response.assessment_label.value
        cat_stats["assessment_labels"][lbl] = cat_stats["assessment_labels"].get(lbl, 0) + 1

        if response.assessment_label == AssessmentLabel.INCONCLUSIVE:
            cat_stats["inconclusive_count"] += 1

        # Binary confusion tracking
        if sample.is_in_domain:
            if predicted_in_domain:
                tp += 1
                cat_stats["correct_domain"] += 1
            else:
                fn += 1
        else:
            # OOD Sample
            if predicted_in_domain:
                fp += 1
            else:
                tn += 1
                cat_stats["correct_domain"] += 1

            # Safety check: OOD samples must NEVER receive a forced binary classification
            if response.assessment_label in {
                AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
                AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
                AssessmentLabel.SUSPICIOUS_MANIPULATION,
            } and response.analysis_eligibility:
                forced_binary_on_ood += 1

            if response.assessment_label == AssessmentLabel.INCONCLUSIVE:
                inconclusive_on_ood += 1

    total = len(dataset)
    in_domain_total = sum(1 for s in dataset if s.is_in_domain)
    ood_total = sum(1 for s in dataset if not s.is_in_domain)

    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-6, (prec + rec))
    far = fp / max(1, ood_total)  # False Acceptance Rate
    frr = fn / max(1, in_domain_total)  # False Rejection Rate
    inconclusive_rate = inconclusive_on_ood / max(1, ood_total)

    return OODEvaluationSummary(
        total_samples=total,
        in_domain_samples=in_domain_total,
        ood_samples=ood_total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1_score=float(f1),
        false_acceptance_rate=float(far),
        false_rejection_rate=float(frr),
        forced_binary_on_ood_count=forced_binary_on_ood,
        inconclusive_on_ood_rate=float(inconclusive_rate),
        per_category_metrics=per_category,
        sha256_invariance_verified=all_sha256_invariant,
    )
