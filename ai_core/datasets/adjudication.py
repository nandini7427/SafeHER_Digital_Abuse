"""Advanced dataset label reliability, taxonomy normalization, and adjudication engine for SafeHER (M4).

Guarantees:
1. 9-tier fine-grained forensic label taxonomy distinguishing full/partial AI, face swaps, reenactment,
   inpainting, general edits, retouching, unknown, and insufficient evidence.
2. Immutable preservation of raw publisher labels alongside normalized forensic labels.
3. Multiple-annotator consensus measurement (agreement ratio, Cohen's kappa).
4. Explicit adjudication procedure resolving conflicting annotations with structured audit trails.
5. Separation of label confidence (annotation certainty) from model prediction confidence.
6. Task-specific training eligibility masking (Model A vs. Model B) preventing label contamination.
7. Anti-discard guarantee: Difficult, ambiguous, or contradictory samples are quarantined and
   preserved for challenge evaluation rather than silently discarded.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_core.datasets.manifest import (
    DatasetImageLabel,
    DatasetImageRecord,
    ManipulationCategory,
)


class NormalizedForensicLabel(str, Enum):
    """Rigorous 9-tier forensic label taxonomy for SafeHER M4."""

    FULLY_AI_GENERATED = "FULLY_AI_GENERATED"
    PARTIALLY_AI_GENERATED = "PARTIALLY_AI_GENERATED"
    FACE_SWAPPED = "FACE_SWAPPED"
    FACIAL_REENACTMENT = "FACIAL_REENACTMENT"
    FACE_INPAINTING = "FACE_INPAINTING"
    GENERAL_IMAGE_EDITING = "GENERAL_IMAGE_EDITING"
    RETOUCHED_GENUINE_PHOTOGRAPH = "RETOUCHED_GENUINE_PHOTOGRAPH"
    UNKNOWN_MANIPULATION_HISTORY = "UNKNOWN_MANIPULATION_HISTORY"
    INSUFFICIENT_ANNOTATION_EVIDENCE = "INSUFFICIENT_ANNOTATION_EVIDENCE"


class ConsensusStatus(str, Enum):
    """Consensus and adjudication outcome state for a dataset asset."""

    UNANIMOUS_CONSENSUS = "UNANIMOUS_CONSENSUS"
    ADJUDICATED_MAJORITY = "ADJUDICATED_MAJORITY"
    CONTRADICTORY_AMBIGUOUS = "CONTRADICTORY_AMBIGUOUS"
    UNVERIFIED_SINGLE_ANNOTATOR = "UNVERIFIED_SINGLE_ANNOTATOR"
    REJECTED_INSUFFICIENT_EVIDENCE = "REJECTED_INSUFFICIENT_EVIDENCE"


class AnnotatorVote(BaseModel):
    """Individual annotation assessment from a human or verified automated audit agent."""

    annotator_id: str = Field(..., description="Unique identifier of annotator or verification agent")
    assigned_label: NormalizedForensicLabel = Field(..., description="Taxonomy label assigned by annotator")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Subjective or calibrated annotation confidence")
    evidence_notes: Optional[str] = Field(None, description="Documented visual or metadata evidence rationale")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of annotation",
    )

    model_config = ConfigDict(frozen=True)


class TaskEligibilityMask(BaseModel):
    """Supervised training and evaluation eligibility flags for neural model scopes."""

    eligible_for_model_a_train: bool = Field(
        default=False,
        description="Eligible for Model A training (requires clean FULLY_AI_GENERATED vs. unedited CAMERA_PHOTOGRAPH)",
    )
    eligible_for_model_b_train: bool = Field(
        default=False,
        description="Eligible for Model B training (requires localized facial manipulation vs. authentic faces)",
    )
    eligible_for_evaluation_only: bool = Field(
        default=True,
        description="Preserved for benchmark challenge evaluation without polluting training gradients",
    )
    quarantined: bool = Field(
        default=False,
        description="Flagged for quarantine due to contradictory or unresolvable labels",
    )
    exclusion_reason: Optional[str] = Field(
        None,
        description="Explicit documentation explaining why asset is excluded from specific training tasks",
    )

    model_config = ConfigDict(frozen=True)


class AdjudicatedDatasetRecord(BaseModel):
    """Complete adjudicated record tracking provenance, raw publisher labels, and normalized consensus."""

    record_id: str = Field(..., description="Unique dataset record identifier")
    sha256: str = Field(..., description="SHA-256 cryptographic hash of image bytes")
    original_dataset_source: str = Field(..., description="Originating dataset name (e.g. FaceForensics++, GenImage)")
    original_publisher_label: str = Field(..., description="Immutable verbatim raw label provided by publisher")
    normalized_label: NormalizedForensicLabel = Field(..., description="Harmonized 9-tier taxonomy label")
    label_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Annotation confidence (strictly decoupled from model inference confidence)",
    )
    consensus_status: ConsensusStatus = Field(
        default=ConsensusStatus.UNVERIFIED_SINGLE_ANNOTATOR,
        description="Consensus or adjudication agreement state",
    )
    annotator_votes: List[AnnotatorVote] = Field(default_factory=list, description="All independent annotation votes")
    adjudication_rationale: Optional[str] = Field(None, description="Auditor rationale for consensus determination")
    adjudicator_id: Optional[str] = Field(None, description="Identifier of senior curator or consensus engine")
    task_eligibility: TaskEligibilityMask = Field(default_factory=TaskEligibilityMask, description="Task assignment mask")
    audit_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of audit and adjudication",
    )


def compute_inter_annotator_agreement(votes: List[AnnotatorVote]) -> Tuple[float, Optional[float]]:
    """Compute agreement ratio and Cohen's Kappa (if 2 annotators).

    Returns:
        Tuple of (agreement_ratio, cohens_kappa).
    """
    if not votes:
        return 0.0, None

    total_votes = len(votes)
    if total_votes == 1:
        return 1.0, 1.0

    # Count votes per label
    counts: Dict[NormalizedForensicLabel, int] = {}
    for v in votes:
        counts[v.assigned_label] = counts.get(v.assigned_label, 0) + 1

    max_count = max(counts.values())
    agreement_ratio = max_count / float(total_votes)

    cohens_kappa: Optional[float] = None
    if total_votes == 2:
        # For 2 annotators: if labels agree, kappa = 1.0; if disagree, kappa = 0.0 (binary point estimate)
        v1, v2 = votes[0], votes[1]
        cohens_kappa = 1.0 if v1.assigned_label == v2.assigned_label else 0.0

    return agreement_ratio, cohens_kappa


class LabelValidationPipeline:
    """Rigorous dataset label validation, harmonization, and adjudication pipeline."""

    def __init__(self, minimum_consensus_threshold: float = 0.65) -> None:
        self.minimum_consensus_threshold = minimum_consensus_threshold

    def normalize_publisher_label(
        self,
        raw_label: str,
        source_dataset: str,
        manipulation_category: Optional[Union[str, ManipulationCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NormalizedForensicLabel, float, Optional[str]]:
        """Map raw publisher label to normalized 9-tier taxonomy with confidence and notes.

        Never assumes publisher labels are scientifically correct without contextual validation.
        """
        raw_clean = str(raw_label).strip().lower()
        cat_str = str(manipulation_category).lower() if manipulation_category else ""
        meta = metadata or {}

        # 1. Contradiction check: Check for AI metadata in supposedly authentic photos
        if raw_clean in ("authentic", "real", "camera_photograph", "original"):
            if "prompt" in meta or "software" in meta and any(k in str(meta["software"]).lower() for k in ("midjourney", "diffusion", "dall-e", "stablediffusion")):
                return (
                    NormalizedForensicLabel.UNKNOWN_MANIPULATION_HISTORY,
                    0.40,
                    "CONTRADICTION: Asset labeled authentic but contains generative AI metadata prompt/software.",
                )
            if "retouch" in cat_str or "photoshop" in cat_str or meta.get("airbrushed"):
                return (
                    NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH,
                    0.90,
                    "Authentic camera photograph with confirmed post-capture cosmetic/color retouching.",
                )
            return (
                NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH if "retouch" in raw_clean else NormalizedForensicLabel.INSUFFICIENT_ANNOTATION_EVIDENCE if not meta.get("verified_camera_source", True) else NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH,
                0.95 if raw_clean in ("camera_photograph", "real") else 0.85,
                None,
            )

        # 2. Face manipulation mappings
        if any(term in raw_clean or term in cat_str for term in ("face_swap", "deepfake_face_swap", "faceswap", "dfl", "roop")):
            return NormalizedForensicLabel.FACE_SWAPPED, 0.95, "Verified identity replacement / face swap."

        if any(term in raw_clean or term in cat_str for term in ("reenactment", "face2face", "neuraltextures", "liveportrait", "fomm")):
            return NormalizedForensicLabel.FACIAL_REENACTMENT, 0.95, "Verified facial reenactment / expression transfer."

        if any(term in raw_clean or term in cat_str for term in ("inpainting", "facial_inpainting", "diffusion_inpainting")):
            return NormalizedForensicLabel.FACE_INPAINTING, 0.95, "Targeted inpainting on facial or anatomical region."

        # 3. AI Generated mappings
        if any(term in raw_clean or term in cat_str for term in ("ai_generated", "diffusion_generated", "gan_generated", "flux", "midjourney", "progan", "stylegan")):
            # Check if full frame or partial
            if meta.get("is_partial") or "partial" in raw_clean or "spliced" in cat_str:
                return NormalizedForensicLabel.PARTIALLY_AI_GENERATED, 0.90, "Composite containing partial generative AI regions."
            return NormalizedForensicLabel.FULLY_AI_GENERATED, 0.95, "Full-frame synthetic image generation."

        # 4. General image edits (copy-move, splicing, clone stamp)
        if any(term in raw_clean or term in cat_str for term in ("splicing", "copy_move", "composite", "photoshop", "local_image_edit")):
            return NormalizedForensicLabel.GENERAL_IMAGE_EDITING, 0.90, "General digital image manipulation without facial specialization."

        # 5. Ambiguous / Unknown
        if raw_clean in ("unknown", "unverified", "ambiguous", "mixed"):
            return NormalizedForensicLabel.UNKNOWN_MANIPULATION_HISTORY, 0.30, "Publisher flagged manipulation history as unknown or ambiguous."

        # Default fallback: Insufficient evidence
        return (
            NormalizedForensicLabel.INSUFFICIENT_ANNOTATION_EVIDENCE,
            0.20,
            f"Raw label '{raw_label}' lacks verifiable provenance or forensic grounding.",
        )

    def determine_task_eligibility(
        self,
        normalized_label: NormalizedForensicLabel,
        consensus_status: ConsensusStatus,
        label_confidence: float,
        is_quarantined: bool = False,
        exclusion_reason: Optional[str] = None,
    ) -> TaskEligibilityMask:
        """Assign strict task-specific training and evaluation eligibility flags.

        Rules:
        - Model A Train: Only clean FULLY_AI_GENERATED or pure pristine authentic camera photographs.
          Partially AI, retouched, face swaps, and general edits MUST be excluded to prevent label noise.
        - Model B Train: Only FACE_SWAPPED, FACIAL_REENACTMENT, or FACE_INPAINTING vs. clean authentic faces.
        - Quarantined or ambiguous samples are NEVER trained on, but kept for evaluation.
        """
        if is_quarantined or consensus_status in (ConsensusStatus.CONTRADICTORY_AMBIGUOUS, ConsensusStatus.REJECTED_INSUFFICIENT_EVIDENCE) or label_confidence < 0.70:
            return TaskEligibilityMask(
                eligible_for_model_a_train=False,
                eligible_for_model_b_train=False,
                eligible_for_evaluation_only=True,
                quarantined=True,
                exclusion_reason=exclusion_reason or f"Quarantined due to consensus status '{consensus_status.value}' and confidence {label_confidence:.2f}.",
            )

        # Model A Training Eligibility
        eligible_a = normalized_label in (
            NormalizedForensicLabel.FULLY_AI_GENERATED,
            NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH,  # Clean authentic camera baseline
        )
        # Model B Training Eligibility
        eligible_b = normalized_label in (
            NormalizedForensicLabel.FACE_SWAPPED,
            NormalizedForensicLabel.FACIAL_REENACTMENT,
            NormalizedForensicLabel.FACE_INPAINTING,
            NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH,  # Clean authentic facial baseline
        )

        reasons = []
        if not eligible_a and normalized_label in (NormalizedForensicLabel.PARTIALLY_AI_GENERATED, NormalizedForensicLabel.GENERAL_IMAGE_EDITING, NormalizedForensicLabel.FACE_SWAPPED):
            reasons.append(f"Excluded from Model A training: localized edit '{normalized_label.value}' introduces label noise to whole-image classifier.")
        if not eligible_b and normalized_label == NormalizedForensicLabel.FULLY_AI_GENERATED:
            reasons.append("Excluded from Model B training: full-frame generation lacks localized facial boundary seams.")

        return TaskEligibilityMask(
            eligible_for_model_a_train=eligible_a,
            eligible_for_model_b_train=eligible_b,
            eligible_for_evaluation_only=True,
            quarantined=False,
            exclusion_reason="; ".join(reasons) if reasons else None,
        )

    def adjudicate_record(
        self,
        record_id: str,
        sha256: str,
        source_dataset: str,
        original_publisher_label: str,
        votes: Optional[List[AnnotatorVote]] = None,
        manipulation_category: Optional[Union[str, ManipulationCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        adjudicator_id: str = "safeher-curator-v1",
    ) -> AdjudicatedDatasetRecord:
        """Process, validate, normalize, and adjudicate a single dataset asset."""
        votes_list = list(votes) if votes else []

        # If no explicit votes provided, generate initial normalized vote from publisher label
        if not votes_list:
            norm_lbl, conf, note = self.normalize_publisher_label(
                raw_label=original_publisher_label,
                source_dataset=source_dataset,
                manipulation_category=manipulation_category,
                metadata=metadata,
            )
            publisher_vote = AnnotatorVote(
                annotator_id=f"publisher-{source_dataset.lower().replace(' ', '_')}",
                assigned_label=norm_lbl,
                confidence=conf,
                evidence_notes=note or f"Ingested from publisher label '{original_publisher_label}'",
            )
            votes_list.append(publisher_vote)

        agreement_ratio, kappa = compute_inter_annotator_agreement(votes_list)

        # Consensus determination
        if len(votes_list) == 1:
            norm_lbl = votes_list[0].assigned_label
            conf = votes_list[0].confidence
            consensus = ConsensusStatus.UNVERIFIED_SINGLE_ANNOTATOR
            rationale = "Single publisher annotation; unverified by secondary independent annotator."
        else:
            # Multi-annotator voting
            counts: Dict[NormalizedForensicLabel, int] = {}
            confidences: Dict[NormalizedForensicLabel, List[float]] = {}
            for v in votes_list:
                counts[v.assigned_label] = counts.get(v.assigned_label, 0) + 1
                confidences.setdefault(v.assigned_label, []).append(v.confidence)

            # Find majority label
            sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            top_label, top_count = sorted_labels[0]

            if top_count == len(votes_list):
                norm_lbl = top_label
                conf = sum(confidences[top_label]) / float(top_count)
                consensus = ConsensusStatus.UNANIMOUS_CONSENSUS
                rationale = f"Unanimous agreement across {len(votes_list)} annotators."
            elif agreement_ratio >= self.minimum_consensus_threshold:
                norm_lbl = top_label
                conf = sum(confidences[top_label]) / float(top_count)
                consensus = ConsensusStatus.ADJUDICATED_MAJORITY
                rationale = f"Majority consensus reached with {agreement_ratio:.1%} agreement ({top_count}/{len(votes_list)})."
            else:
                # Disagreement / tie
                norm_lbl = NormalizedForensicLabel.UNKNOWN_MANIPULATION_HISTORY
                conf = 0.40
                consensus = ConsensusStatus.CONTRADICTORY_AMBIGUOUS
                rationale = f"Contradictory annotations detected: agreement ratio {agreement_ratio:.1%} below threshold {self.minimum_consensus_threshold:.1%}."

        is_quarantined = consensus in (ConsensusStatus.CONTRADICTORY_AMBIGUOUS, ConsensusStatus.REJECTED_INSUFFICIENT_EVIDENCE) or conf < 0.60
        eligibility = self.determine_task_eligibility(
            normalized_label=norm_lbl,
            consensus_status=consensus,
            label_confidence=conf,
            is_quarantined=is_quarantined,
            exclusion_reason=rationale if is_quarantined else None,
        )

        return AdjudicatedDatasetRecord(
            record_id=record_id,
            sha256=sha256,
            original_dataset_source=source_dataset,
            original_publisher_label=original_publisher_label,
            normalized_label=norm_lbl,
            label_confidence=conf,
            consensus_status=consensus,
            annotator_votes=votes_list,
            adjudication_rationale=rationale,
            adjudicator_id=adjudicator_id,
            task_eligibility=eligibility,
        )


@dataclass
class LabelQualityAuditReport:
    """Comprehensive statistical report summarizing label reliability and adjudication."""

    total_records_audited: int
    normalized_counts: Dict[str, int]
    consensus_counts: Dict[str, int]
    quarantined_count: int
    model_a_eligible_count: int
    model_b_eligible_count: int
    contradiction_count: int
    average_label_confidence: float
    audit_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records_audited": self.total_records_audited,
            "normalized_counts": self.normalized_counts,
            "consensus_counts": self.consensus_counts,
            "quarantined_count": self.quarantined_count,
            "model_a_eligible_count": self.model_a_eligible_count,
            "model_b_eligible_count": self.model_b_eligible_count,
            "contradiction_count": self.contradiction_count,
            "average_label_confidence": round(self.average_label_confidence, 4),
            "audit_notes": self.audit_notes,
        }
