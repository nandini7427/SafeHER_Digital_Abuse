"""Explainability engine, uncertainty assessment, and safety bounds for SafeHER Image Intelligence."""

from typing import List, Optional, Tuple

from ai_core.image_intelligence.preprocessing import PreprocessedImage
from ai_core.image_intelligence.quality import evaluate_image_quality
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ManipulationRegion,
    UncertaintyState,
)


STANDARD_MODEL_LIMITATIONS: List[str] = [
    "A negative detection result indicates the absence of detected manipulation signatures; it does NOT prove an image is authentic or unaltered.",
    "System Scope: SafeHER's default pipeline operates as an image-forensics anomaly triage tool, not a trained neural deepfake classifier.",
    "ELA Limitations: Error Level Analysis detects compression rate discrepancies; however, benign resizing, format conversions (e.g. PNG to JPEG), or multi-source camera stitching can trigger elevated ELA error without malicious tampering.",
    "FFT Limitations: 2D Fourier power spectra are influenced by camera ISP pipelines, sensor sharpening, and denoising; spectral decay slope (alpha ~ 2.0) is a general statistical tendency, not a universal physical law.",
    "Model Non-Independence: ELA and 2D-FFT analyses inspect transformations of the exact same pixel matrix; their agreement provides corroborative consistency, not statistically independent confirmation.",
    "Automated detection heuristics are sensitive to social-media platform recompression (e.g. WhatsApp, Instagram, Telegram), which destroys high-frequency noise and compression tables.",
    "The detector cannot assess the real-world identity, intent, or consent of any individual depicted in the imagery.",
    "Cosmetic filters (such as beauty smoothing or background blurs) and phone screenshot compositors can trigger localized false positives.",
    "Evidence Preservation: Images rejected as unsupported domains (screenshots, documents, memes) are securely preserved in the vault; automated triage ineligibility does not imply the material is benign or unimportant.",
    "This automated triage report does not constitute sworn forensic testimony or legal evidence in court proceedings.",
]


def determine_assessment_label_and_uncertainty(
    raw_score: float,
    preprocessed: PreprocessedImage,
    manipulation_regions: List[ManipulationRegion],
    face_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    model_failed: bool = False,
) -> Tuple[AssessmentLabel, UncertaintyState]:
    """Map raw scores and image characteristics to qualitative assessment labels and uncertainty states."""
    width, height = preprocessed.metadata.dimensions

    # 1. Evaluate image quality using the forensic quality engine
    quality_res = evaluate_image_quality(
        img=preprocessed.working_image,
        raw_score=raw_score,
        face_bboxes=face_bboxes,
        model_failed=model_failed,
        format_name=preprocessed.metadata.format,
        has_exif=preprocessed.metadata.has_exif,
    )

    if not quality_res.analysis_eligibility:
        return quality_res.assessment_label, quality_res.uncertainty_state

    # 2. Conflicting spatial signals triage
    if manipulation_regions and raw_score < 0.35:
        # Conflicting signals: localized anomalies exist but overall score is low
        return AssessmentLabel.SUSPICIOUS_MANIPULATION, UncertaintyState.MODEL_UNCERTAIN

    return quality_res.assessment_label, quality_res.uncertainty_state


def build_forensic_explanation(
    detector_name: str,
    raw_score: float,
    label: AssessmentLabel,
    uncertainty: UncertaintyState,
    manipulation_regions: List[ManipulationRegion],
    preprocessed: PreprocessedImage,
    auxiliary_metrics: dict,
    quality_warnings: Optional[List[str]] = None,
    reason_for_insufficient_evidence: Optional[str] = None,
) -> str:
    """Generate an explainable, factually grounded forensic summary without fabricating claims.

    Explains:
    1. What the detector examined.
    2. What visual evidence was identified.
    3. Whether spatial localization is supported.
    4. What the model cannot determine.
    5. Why human forensic verification is required.
    """
    dimensions_str = f"{preprocessed.metadata.dimensions[0]}x{preprocessed.metadata.dimensions[1]}"
    format_str = preprocessed.metadata.format

    parts: List[str] = []

    # 1. Scope of examination
    parts.append(
        f"The {detector_name} evaluated this {format_str} image ({dimensions_str} px) "
        "by inspecting pixel-level Error Level Analysis (ELA) compression residuals, "
        "high-frequency boundary gradients, and spatial noise consistency."
    )

    # 2. Evidence identified
    if label in (AssessmentLabel.HIGHLY_LIKELY_MANIPULATED, AssessmentLabel.SUSPICIOUS_MANIPULATION):
        if manipulation_regions:
            quadrant_count = len(manipulation_regions)
            parts.append(
                f"Visual anomaly scan identified {quadrant_count} spatial grid region(s) displaying "
                f"statistically anomalous re-compression variance (mean diff: {auxiliary_metrics.get('ela_mean_diff', 'N/A')}, "
                f"variance: {auxiliary_metrics.get('ela_variance', 'N/A')}). These discrepancies suggest localized "
                "digital splicing, multi-source compositing, or selective editing."
            )
        else:
            parts.append(
                f"Global compression analysis produced an elevated anomaly indicator ({raw_score:.2f}), "
                "indicating significant compression rate inconsistency across the image frame."
            )
    elif label == AssessmentLabel.INCONCLUSIVE:
        if reason_for_insufficient_evidence:
            parts.append(
                f"The image resolution ({dimensions_str} px) is too low for reliable micro-forensic inspection "
                f"({reason_for_insufficient_evidence}). Fine-grained compression artifacts cannot be definitively analyzed at this scale."
            )
        else:
            parts.append(
                f"The image resolution ({dimensions_str} px) is too low for reliable micro-forensic inspection. "
                "Fine-grained compression artifacts cannot be definitively analyzed at this scale."
            )
    else:
        parts.append(
            f"No statistically significant compression inconsistencies or splicing boundary anomalies "
            f"were detected across the image grid (raw indicator: {raw_score:.2f}). Note: This finding "
            "does NOT confirm that the image is authentic; sophisticated generative models and recompressed "
            "files may leave no detectable compression artifacts."
        )

    # 3. Localization support status
    if manipulation_regions:
        parts.append(
            f"Spatial localization is supported via {len(manipulation_regions)} flagged bounding grid cells. "
            "Users can inspect the highlighted coordinates in the UI overlay to examine specific anomaly clusters."
        )
    else:
        parts.append(
            "Spatial anomaly localization was not triggered or is not supported for this level of signal distribution."
        )

    # 4. Uncertainty & safety boundary
    if uncertainty in (UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE, UncertaintyState.INSUFFICIENT_QUALITY):
        parts.append(
            "Uncertainty is categorized as INSUFFICIENT_QUALITY / POOR_QUALITY_EVIDENCE due to low image dimensions, "
            "blur, severe compression, or lighting degradation."
        )
    elif uncertainty in (UncertaintyState.MODEL_UNCERTAIN, UncertaintyState.MODERATE):
        parts.append(
            "Uncertainty is categorized as MODERATE / MODEL_UNCERTAIN because detection signals hover near the decision boundary."
        )
    elif uncertainty == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN:
        parts.append(
            "Uncertainty is categorized as OUTSIDE_SUPPORTED_DOMAIN; the image lacks natural photographic camera textures."
        )
    elif uncertainty == UncertaintyState.MODEL_UNAVAILABLE:
        parts.append(
            "Uncertainty is categorized as MODEL_UNAVAILABLE; the primary detection runner was offline or unavailable."
        )
    elif uncertainty == UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED:
        parts.append(
            "Uncertainty is categorized as ANALYSIS_CANNOT_BE_COMPLETED due to stream corruption or decoding failure."
        )
    elif uncertainty in (UncertaintyState.ANALYSIS_USABLE_EVIDENCE, UncertaintyState.LOW):
        parts.append(
            "Uncertainty is categorized as LOW; evidence meets micro-forensic resolution and quality requirements."
        )

    # 5. Requirement for human verification
    parts.append(
        "Automated heuristics cannot determine identity, consent, or the situational authenticity of depicted scenes. "
        "Independent verification by a qualified digital forensic analyst is required before making legal or protective decisions."
    )

    return " ".join(parts)
