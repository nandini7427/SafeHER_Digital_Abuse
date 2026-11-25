"""SafeHER Out-of-Distribution (OOD) and Domain Suitability Assessment Engine.

Determines whether an uploaded image belongs to the validated photographic domain
on which SafeHER's deepfake and image manipulation models were trained and calibrated.

Supported In-Domain Categories:
- Photographic Portraits / Intimate Evidence (natural camera sensors with detected human subjects)
- Photographic Scenes (continuous tones, optical point-spread function, natural sensor noise)

Unsupported Out-of-Distribution (OOD) Categories:
- Drawings / Pencil Sketches / Anime
- Digital Illustrations / Cartoons / Vector Art
- Memes / Viral Text Overlay Graphics
- Screenshots / Mobile UI Grabs / Chat Logs
- Medical Imagery (X-Ray / MRI / CT / Ultrasound)
- Documents / Scanned Invoices / Contracts / Text
- Landscapes / Scenery without Human Subjects (when face triage is required)
- Extremely Compressed Images (obscuring forensic sensor PRNU)
- Unsupported / Corrupted Image Formats

Guarantees:
- Avoids forced binary classification on unsupported domains (strictly returns INCONCLUSIVE).
- Returns explicit insufficient-evidence triage state (OUTSIDE_SUPPORTED_DOMAIN).
- Preserves original cryptographic evidence invariance (SHA-256).
- Privacy-safe logging without victim image content leakage.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    UncertaintyState,
)

logger = logging.getLogger("safeher.image_intelligence.ood")


class DomainCategory(str, Enum):
    """Categorical classification of image evidence domain."""

    # Supported In-Domain Categories
    PHOTOGRAPHIC_PORTRAIT = "photographic_portrait"
    PHOTOGRAPHIC_SCENE = "photographic_scene"

    # Unsupported Out-of-Distribution (OOD) Categories
    DRAWING_OR_SKETCH = "drawing_or_sketch"
    DIGITAL_ILLUSTRATION = "digital_illustration"
    MEME_OR_TEXT_GRAPHIC = "meme_or_text_graphic"
    SCREENSHOT_OR_UI = "screenshot_or_ui"
    MEDICAL_IMAGERY = "medical_imagery"
    DOCUMENT_OR_SCANNED_TEXT = "document_or_scanned_text"
    LANDSCAPE_WITHOUT_SUBJECT = "landscape_without_subject"
    FACE_NOT_DETECTED = "face_not_detected"
    EXTREMELY_COMPRESSED = "extremely_compressed"
    UNSUPPORTED_FORMAT = "unsupported_format"


SUPPORTED_DOMAINS = {
    DomainCategory.PHOTOGRAPHIC_PORTRAIT,
    DomainCategory.PHOTOGRAPHIC_SCENE,
}

UNSUPPORTED_DOMAINS = {
    DomainCategory.DRAWING_OR_SKETCH,
    DomainCategory.DIGITAL_ILLUSTRATION,
    DomainCategory.MEME_OR_TEXT_GRAPHIC,
    DomainCategory.SCREENSHOT_OR_UI,
    DomainCategory.MEDICAL_IMAGERY,
    DomainCategory.DOCUMENT_OR_SCANNED_TEXT,
    DomainCategory.LANDSCAPE_WITHOUT_SUBJECT,
    DomainCategory.FACE_NOT_DETECTED,
    DomainCategory.EXTREMELY_COMPRESSED,
    DomainCategory.UNSUPPORTED_FORMAT,
}


@dataclass(frozen=True)
class DomainAssessmentResult:
    """Forensic domain assessment result with metrics, warnings, and safe triage."""

    is_in_domain: bool
    category: DomainCategory
    confidence: float
    reason: Optional[str]
    warnings: List[str]
    uncertainty_state: UncertaintyState
    analysis_eligibility: bool
    recommended_action: str
    assessment_label: AssessmentLabel
    technical_metrics: Dict[str, float] = field(default_factory=dict)


def compute_compression_blockiness(gray_arr: np.ndarray) -> float:
    """Measure the ratio of 8x8 DCT grid boundary discontinuities vs intra-block differences."""
    h, w = gray_arr.shape
    if h < 24 or w < 24:
        return 1.0

    diffs_x = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
    boundary_cols = [c for c in range(7, w - 1, 8)]
    interior_cols = [c for c in range(w - 1) if c not in boundary_cols]

    if not boundary_cols or not interior_cols:
        return 1.0

    boundary_mean_x = float(np.mean(diffs_x[:, boundary_cols]))
    interior_mean_x = float(np.mean(diffs_x[:, interior_cols]))

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


class OutOfDistributionDetector:
    """Heuristic and statistical domain classifier for visual evidence triage.

    Evaluates color entropy, palette quantization, spatial gradient distributions,
    border/perimeter profiles, text frequency characteristics, and subject presence.
    """

    # Common display resolutions for mobile and desktop screenshots
    COMMON_SCREEN_RESOLUTIONS = {
        (1080, 1920), (1920, 1080),
        (1080, 2340), (2340, 1080),
        (1080, 2400), (2400, 1080),
        (1170, 2532), (2532, 1170),
        (1284, 2778), (2778, 1284),
        (1179, 2556), (2556, 1179),
        (1290, 2796), (2796, 1290),
        (1440, 2560), (2560, 1440),
        (1440, 3120), (3120, 1440),
        (1440, 3200), (3200, 1440),
    }

    def __init__(self, require_face_for_face_tasks: bool = True) -> None:
        self.require_face_for_face_tasks = require_face_for_face_tasks

    def extract_domain_features(self, img: Image.Image) -> Dict[str, Any]:
        """Extract statistical, chromatic, and gradient features for domain evaluation."""
        width, height = img.size
        aspect_ratio = float(width / max(1, height))

        rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
        h, w, _ = rgb_arr.shape

        # Grayscale ITU-R BT.601
        gray_arr = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]

        # 1. Chromatic Saturation (HSV S)
        max_c = np.maximum(np.maximum(rgb_arr[:, :, 0], rgb_arr[:, :, 1]), rgb_arr[:, :, 2])
        min_c = np.minimum(np.minimum(rgb_arr[:, :, 0], rgb_arr[:, :, 1]), rgb_arr[:, :, 2])
        denom = np.where(max_c > 1.0, max_c, 1.0)
        saturation = np.where(max_c > 1.0, (max_c - min_c) / denom, 0.0)
        mean_saturation = float(np.mean(saturation))
        max_saturation = float(np.max(saturation))

        # 2. Luminance distribution
        mean_luminance = float(np.mean(gray_arr))
        white_ratio = float(np.mean(gray_arr >= 225.0))
        dark_ratio = float(np.mean(gray_arr <= 50.0))
        mid_ratio = float(np.mean((gray_arr > 80.0) & (gray_arr < 225.0)))

        # 3. Outer boundary / perimeter black ratio (typical in medical radiograms)
        border_h = max(1, int(h * 0.05))
        border_w = max(1, int(w * 0.05))
        b_top = gray_arr[:border_h, :]
        b_bottom = gray_arr[-border_h:, :]
        b_left = gray_arr[:, :border_w]
        b_right = gray_arr[:, -border_w:]
        border_pixels = np.concatenate([b_top.ravel(), b_bottom.ravel(), b_left.ravel(), b_right.ravel()])
        border_black_ratio = float(np.mean(border_pixels <= 25.0))

        # 4. Spatial Gradients & Edge Density
        if h > 2 and w > 2:
            grad_x = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
            grad_y = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])
            edge_map = (grad_x[:-1, :] + grad_y[:, :-1]) > 35.0
            edge_density = float(np.mean(edge_map))
            flat_map = (grad_x[:-1, :] + grad_y[:, :-1]) < 2.0
            flat_ratio = float(np.mean(flat_map))
            horizontal_text_lines = float(np.mean(grad_x > 45.0))
        else:
            edge_density = 0.0
            flat_ratio = 1.0
            horizontal_text_lines = 0.0

        # 5. Color Palette Quantization & Shannon Entropy (16x16x16 bins = 4096)
        r_q = (rgb_arr[:, :, 0] / 16).astype(np.int32).clip(0, 15)
        g_q = (rgb_arr[:, :, 1] / 16).astype(np.int32).clip(0, 15)
        b_q = (rgb_arr[:, :, 2] / 16).astype(np.int32).clip(0, 15)
        color_bins = r_q * 256 + g_q * 16 + b_q
        bin_counts = np.bincount(color_bins.ravel(), minlength=4096)
        non_zero_counts = bin_counts[bin_counts > 0]
        total_pixels = float(h * w)
        probs = non_zero_counts / total_pixels
        color_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        top5_color_ratio = float(np.sum(np.sort(bin_counts)[-5:]) / total_pixels)
        unique_colors_4096 = int(np.count_nonzero(bin_counts))

        # 6. Banner / Caption Text Detection (top 15% or bottom 15% strips)
        strip_h = max(2, int(h * 0.15))
        top_strip = gray_arr[:strip_h, :]
        bottom_strip = gray_arr[-strip_h:, :]

        top_edges = (
            np.abs(top_strip[:, 1:] - top_strip[:, :-1])[:-1, :]
            + np.abs(top_strip[1:, :] - top_strip[:-1, :])[:, :-1]
        ) > 30.0
        bottom_edges = (
            np.abs(bottom_strip[:, 1:] - bottom_strip[:, :-1])[:-1, :]
            + np.abs(bottom_strip[1:, :] - bottom_strip[:-1, :])[:, :-1]
        ) > 30.0

        top_is_banner = (
            (float(np.mean(top_strip > 200.0)) + float(np.mean(top_strip < 60.0))) > 0.85
            and float(np.mean(top_edges)) > 0.04
        )
        bottom_is_banner = (
            (float(np.mean(bottom_strip > 200.0)) + float(np.mean(bottom_strip < 60.0))) > 0.85
            and float(np.mean(bottom_edges)) > 0.04
        )
        has_text_banner = (top_is_banner or bottom_is_banner) and (white_ratio < 0.50)

        # 7. UI Status Bar check (top 3% uniform bar)
        status_bar_h = max(1, int(h * 0.035))
        status_bar_strip = gray_arr[:status_bar_h, :]
        status_bar_var = float(np.var(status_bar_strip))

        # 8. Human Subject & Landscape Chromatic Indicators
        r_c = rgb_arr[:, :, 0]
        g_c = rgb_arr[:, :, 1]
        b_c = rgb_arr[:, :, 2]
        skin_mask = (
            (r_c >= 75.0)
            & (g_c >= 35.0)
            & (b_c >= 20.0)
            & (r_c > g_c)
            & (g_c > b_c)
            & ((r_c - g_c) >= 10.0)
            & ((r_c - b_c) >= 15.0)
        )
        skin_ratio = float(np.mean(skin_mask))

        green_mask = (g_c > r_c * 1.05) & (g_c > b_c * 1.05) & (g_c >= 40.0)
        foliage_ratio = float(np.mean(green_mask))

        blue_mask = (b_c > r_c * 1.08) & (b_c > g_c * 1.02) & (b_c >= 80.0)
        sky_ratio = float(np.mean(blue_mask))

        # 9. Text Line Transitions (Document detection)
        dark_row_mask = (np.mean(gray_arr <= 100.0, axis=1) > 0.03)
        text_line_transitions = int(np.sum(dark_row_mask[1:] != dark_row_mask[:-1]))

        # 10. Compression Blockiness
        blockiness = compute_compression_blockiness(gray_arr)

        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "mean_saturation": mean_saturation,
            "max_saturation": max_saturation,
            "mean_luminance": mean_luminance,
            "white_ratio": white_ratio,
            "dark_ratio": dark_ratio,
            "mid_ratio": mid_ratio,
            "border_black_ratio": border_black_ratio,
            "edge_density": edge_density,
            "flat_ratio": flat_ratio,
            "horizontal_text_lines": horizontal_text_lines,
            "color_entropy": color_entropy,
            "top5_color_ratio": top5_color_ratio,
            "unique_colors_4096": unique_colors_4096,
            "has_text_banner": has_text_banner,
            "status_bar_var": status_bar_var,
            "skin_ratio": skin_ratio,
            "foliage_ratio": foliage_ratio,
            "sky_ratio": sky_ratio,
            "text_line_transitions": text_line_transitions,
            "compression_blockiness": blockiness,
        }

    def classify_domain(
        self,
        img: Image.Image,
        format_name: Optional[str] = None,
        has_exif: bool = False,
        task: DetectionTask = DetectionTask.HYBRID_MANIPULATION,
        face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
        is_corrupted: bool = False,
        compression_blockiness: Optional[float] = None,
    ) -> DomainAssessmentResult:
        """Classify the evidence into supported in-domain or unsupported out-of-distribution category."""
        # 1. Unsupported / Corrupted Format
        if is_corrupted or (format_name and format_name.upper() not in {"JPEG", "JPG", "PNG", "WEBP"}):
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.UNSUPPORTED_FORMAT,
                confidence=1.0,
                reason="File format is unsupported or contains a corrupted stream.",
                warnings=["Unsupported file format or corrupted payload detected; cannot be forensically parsed."],
                uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED,
                analysis_eligibility=False,
                recommended_action="Submit valid photographic evidence in standard JPEG, PNG, or WebP format.",
                assessment_label=AssessmentLabel.INCONCLUSIVE,
            )

        feats = self.extract_domain_features(img)
        w = feats["width"]
        h = feats["height"]
        aspect = feats["aspect_ratio"]
        blockiness = compression_blockiness if compression_blockiness is not None else feats["compression_blockiness"]

        # 2. Extreme Compression Artifacts
        if blockiness > 2.5:
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.EXTREMELY_COMPRESSED,
                confidence=0.95,
                reason="Severe 8x8 DCT compression grid blocking destroys sensor noise and boundary forensics.",
                warnings=[
                    f"Extreme compression detected (DCT blockiness: {blockiness:.2f} > 2.5); "
                    "lossy re-quantization prevents reliable deepfake triage."
                ],
                uncertainty_state=UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
                analysis_eligibility=False,
                recommended_action="Request original, uncompressed camera image directly from source storage.",
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 3. Screenshots / UI Grabs
        # Evaluated early so light mobile app backgrounds are not misclassified as documents
        is_screen_dim = (w, h) in self.COMMON_SCREEN_RESOLUTIONS
        is_screen_aspect = 1.95 <= aspect <= 2.25 or 0.43 <= aspect <= 0.52
        is_screen_status = feats["status_bar_var"] < 12.0 and (is_screen_aspect or is_screen_dim)

        if (is_screen_dim or (is_screen_aspect and is_screen_status)) and not has_exif:
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.SCREENSHOT_OR_UI,
                confidence=0.90,
                reason="Image is a mobile device or desktop screen capture rather than a direct camera file.",
                warnings=[
                    "Screen capture / mobile UI detected; display rendering and compositor scaling "
                    "strip native camera sensor PRNU and optical lens characteristics."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action=(
                    "Evidence is preserved in the secure vault. Because automated forensic models require direct camera "
                    "sensor characteristics, screenshots cannot be triaged by automated heuristics. We recommend requesting "
                    "human case review for this evidence, and submitting the uncropped original camera file if available."
                ),
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 4. Medical Imagery (X-Rays, MRI, CT, Ultrasound)
        # Signature: near-zero chrominance (monochromatic), high black border ratio
        if feats["mean_saturation"] <= 0.04 and feats["border_black_ratio"] >= 0.60 and feats["mean_luminance"] >= 15.0:
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.MEDICAL_IMAGERY,
                confidence=0.95,
                reason="Image exhibits diagnostic medical radiogram characteristics (monochromatic with black perimeter framing).",
                warnings=[
                    "Medical radiography characteristics detected (zero chromatic saturation with perimeter masking); "
                    "SafeHER models are not validated for clinical or radiologic imaging."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action="Submit standard optical camera evidence. Medical imaging requires clinical PACS verification.",
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 5. Memes / Viral Text Overlay Graphics
        # Evaluated before digital illustration so banner-framed photos are categorized as memes
        if feats["has_text_banner"]:
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.MEME_OR_TEXT_GRAPHIC,
                confidence=0.85,
                reason="Image contains synthetic meme text banners or rasterized graphic overlays.",
                warnings=[
                    "Synthetic meme text banner or caption overlay detected; digital text re-rasterization "
                    "interferes with optical camera forensic integrity."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action=(
                    "Evidence is preserved in the secure vault. Meme graphics and text banners alter forensic compression tables; "
                    "request human case review for harassment assessment, or submit the original unedited photo if available."
                ),
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 6. Documents / Scanned Text / Receipts
        # Signature: high white background, low mid-tones, presence of dark text lines with alternating row transitions
        if (
            feats["white_ratio"] >= 0.58
            and feats["mid_ratio"] <= 0.22
            and feats["dark_ratio"] >= 0.02
            and feats["text_line_transitions"] >= 16
        ):
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.DOCUMENT_OR_SCANNED_TEXT,
                confidence=0.95,
                reason="Image is a scanned document, text sheet, or receipt, not a camera photograph.",
                warnings=[
                    f"Document/text characteristics detected (white background ratio: {feats['white_ratio']:.1%}, "
                    f"mid-tones: {feats['mid_ratio']:.1%}); outside photographic deepfake domain."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action=(
                    "SafeHER deepfake detection models are validated strictly on photographic evidence. "
                    "For document tampering, submit to a document forensics specialist."
                ),
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 7. Drawings / Line Art / Pencil Sketches
        # Signature: very low saturation, light paper background, high edge density, low unique colors
        if (
            feats["mean_saturation"] <= 0.12
            and feats["white_ratio"] >= 0.35
            and feats["edge_density"] >= 0.03
            and feats["unique_colors_4096"] < 90
        ):
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.DRAWING_OR_SKETCH,
                confidence=0.90,
                reason="Image is a hand-drawn sketch, pencil line art, or monochrome illustration.",
                warnings=[
                    f"Pencil sketch/line art detected (edge density: {feats['edge_density']:.2f}, "
                    f"saturation: {feats['mean_saturation']:.2f}); lacks photographic camera sensor texture."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action=(
                    "Evidence is preserved in the secure vault. Hand-drawn sketches lack physical optical camera textures; "
                    "request manual case review for non-photographic abuse material."
                ),
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 8. Digital Illustrations / Cartoons / Vector Art
        # Signature: high saturation with low color entropy, high top-5 color concentration, high flat ratio
        if (
            feats["color_entropy"] < 4.0
            and feats["top5_color_ratio"] >= 0.55
            and feats["flat_ratio"] >= 0.35
            and feats["mean_saturation"] >= 0.35
            and feats["unique_colors_4096"] <= 60
        ):
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.DIGITAL_ILLUSTRATION,
                confidence=0.90,
                reason="Image is a digital illustration, cartoon, or vector artwork with posterized color fills.",
                warnings=[
                    f"Digital illustration/vector art detected (color entropy: {feats['color_entropy']:.2f} < 4.0, "
                    f"top-5 color ratio: {feats['top5_color_ratio']:.1%}); outside photographic domain."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action=(
                    "Evidence is preserved in the secure vault. Vector illustrations and digital cartoons fall outside the "
                    "validated photographic domain; request human casework review if this graphic is used in targeted harassment."
                ),
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 9. Landscapes / Scenery without Subject
        face_count = len(face_bboxes) if face_bboxes is not None else None
        nature_ratio = feats["foliage_ratio"] + feats["sky_ratio"]
        has_face = (face_count is not None and face_count >= 1)
        has_human_subject = has_face or (face_count is None and feats["skin_ratio"] >= 0.04 and nature_ratio < 0.50)
        is_landscape = (feats["foliage_ratio"] >= 0.08 and feats["sky_ratio"] >= 0.12) and not has_face

        if (not has_human_subject and is_landscape) or (task == DetectionTask.FACE_MANIPULATION and face_count == 0):
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.LANDSCAPE_WITHOUT_SUBJECT,
                confidence=0.90,
                reason="Image is an outdoor landscape/scenery without human subjects; outside validated facial deepfake domain.",
                warnings=[
                    "Landscape/scenery detected without human subjects; SafeHER visual models are trained "
                    "and calibrated specifically for human face manipulation and intimate imagery protection."
                ],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action="Submit photographic evidence containing the human subject or victim for deepfake assessment.",
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        if not has_human_subject and feats["skin_ratio"] < 0.008 and face_count == 0:
            return DomainAssessmentResult(
                is_in_domain=False,
                category=DomainCategory.FACE_NOT_DETECTED,
                confidence=0.85,
                reason="No human face or subject detected in the image.",
                warnings=["No human subject or facial region identified; outside validated portrait triage domain."],
                uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
                analysis_eligibility=False,
                recommended_action="Submit images containing the victim or person of interest for face manipulation analysis.",
                assessment_label=AssessmentLabel.INCONCLUSIVE,
                technical_metrics=feats,
            )

        # 10. Supported In-Domain Classification
        # High entropy, natural continuous photographic gradients
        if has_human_subject or (face_count is not None and face_count >= 1):
            category = DomainCategory.PHOTOGRAPHIC_PORTRAIT
            confidence = 0.95
        else:
            category = DomainCategory.PHOTOGRAPHIC_SCENE
            confidence = 0.90

        return DomainAssessmentResult(
            is_in_domain=True,
            category=category,
            confidence=confidence,
            reason=None,
            warnings=[],
            uncertainty_state=UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
            analysis_eligibility=True,
            recommended_action="Evidence is in the validated photographic domain. Maintain SHA-256 chain of custody.",
            assessment_label=AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
            technical_metrics=feats,
        )
