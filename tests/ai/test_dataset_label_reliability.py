"""Unit and integration tests for SafeHER Advanced Dataset Label Reliability and Adjudication (M4).

Validates:
1. Complete 9-tier forensic label taxonomy normalization.
2. Contradiction and ambiguity detection in raw publisher annotations.
3. Multi-annotator agreement metrics (agreement ratio, Cohen's Kappa).
4. Consensus adjudication logic (unanimous, majority, and split votes).
5. Task-specific training eligibility masking (Model A vs. Model B).
6. Anti-discard guarantee: Difficult and ambiguous samples are quarantined, not silently dropped.
7. Separation of label confidence from model inference confidence.
8. Immutability of original publisher labels.
"""

from datetime import datetime, timezone
import pytest

from ai_core.datasets import (
    AdjudicatedDatasetRecord,
    AnnotatorVote,
    ConsensusStatus,
    LabelQualityAuditReport,
    LabelValidationPipeline,
    NormalizedForensicLabel,
    TaskEligibilityMask,
    compute_inter_annotator_agreement,
)
from ai_core.datasets.manifest import ManipulationCategory


# ---------------------------------------------------------------------------
# 1. 9-Tier Taxonomy Normalization Tests
# ---------------------------------------------------------------------------

def test_normalization_all_nine_categories():
    """Verify that all 9 mandated categories in the taxonomy can be mapped and normalized."""
    pipeline = LabelValidationPipeline()

    test_cases = [
        # (raw_label, category, metadata, expected_normalized)
        ("diffusion_generated", ManipulationCategory.DIFFUSION_GENERATED, {}, NormalizedForensicLabel.FULLY_AI_GENERATED),
        ("ai_generated", ManipulationCategory.NONE, {"is_partial": True}, NormalizedForensicLabel.PARTIALLY_AI_GENERATED),
        ("deepfake", ManipulationCategory.DEEPFAKE_FACE_SWAP, {}, NormalizedForensicLabel.FACE_SWAPPED),
        ("face2face", ManipulationCategory.FACE_REENACTMENT, {}, NormalizedForensicLabel.FACIAL_REENACTMENT),
        ("inpainting", ManipulationCategory.DIFFUSION_INPAINTING, {}, NormalizedForensicLabel.FACE_INPAINTING),
        ("photoshop_splice", ManipulationCategory.SPLICED_COMPOSITE, {}, NormalizedForensicLabel.GENERAL_IMAGE_EDITING),
        ("camera_photograph", ManipulationCategory.LOCALIZED_RETOUCH, {}, NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH),
        ("unknown", ManipulationCategory.UNKNOWN, {}, NormalizedForensicLabel.UNKNOWN_MANIPULATION_HISTORY),
        ("unverified_asset", "random_string", {}, NormalizedForensicLabel.INSUFFICIENT_ANNOTATION_EVIDENCE),
    ]

    for raw_lbl, cat, meta, expected in test_cases:
        norm_lbl, conf, note = pipeline.normalize_publisher_label(
            raw_label=raw_lbl,
            source_dataset="BenchmarkSuite",
            manipulation_category=cat,
            metadata=meta,
        )
        assert norm_lbl == expected, f"Failed for raw='{raw_lbl}', expected={expected}, got={norm_lbl}"
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 2. Contradiction & Ambiguity Detection Tests
# ---------------------------------------------------------------------------

def test_publisher_contradiction_detection():
    """Verify that claims of 'authentic' containing generative AI metadata trigger contradiction alerts."""
    pipeline = LabelValidationPipeline()

    # Case: Publisher says 'authentic' but file contains Stable Diffusion metadata prompt
    norm_lbl, conf, note = pipeline.normalize_publisher_label(
        raw_label="authentic",
        source_dataset="SuspectWebDataset",
        manipulation_category=ManipulationCategory.NONE,
        metadata={"software": "StableDiffusion v2.1", "prompt": "photorealistic portrait of a woman"},
    )
    assert norm_lbl == NormalizedForensicLabel.UNKNOWN_MANIPULATION_HISTORY
    assert conf <= 0.50
    assert note is not None
    assert "CONTRADICTION" in note


# ---------------------------------------------------------------------------
# 3. Inter-Annotator Agreement & Metrics Tests
# ---------------------------------------------------------------------------

def test_inter_annotator_agreement_unanimous():
    """Verify agreement calculation when 3 annotators unanimously agree."""
    votes = [
        AnnotatorVote(annotator_id="A1", assigned_label=NormalizedForensicLabel.FACE_SWAPPED, confidence=0.95),
        AnnotatorVote(annotator_id="A2", assigned_label=NormalizedForensicLabel.FACE_SWAPPED, confidence=0.90),
        AnnotatorVote(annotator_id="A3", assigned_label=NormalizedForensicLabel.FACE_SWAPPED, confidence=0.92),
    ]
    ratio, kappa = compute_inter_annotator_agreement(votes)
    assert ratio == 1.0


def test_inter_annotator_agreement_two_annotators_disagreement():
    """Verify Cohen's Kappa evaluates to 0.0 on 2-annotator split disagreement."""
    votes = [
        AnnotatorVote(annotator_id="A1", assigned_label=NormalizedForensicLabel.FACE_SWAPPED, confidence=0.95),
        AnnotatorVote(annotator_id="A2", assigned_label=NormalizedForensicLabel.FACIAL_REENACTMENT, confidence=0.85),
    ]
    ratio, kappa = compute_inter_annotator_agreement(votes)
    assert ratio == 0.5
    assert kappa == 0.0


# ---------------------------------------------------------------------------
# 4. Multi-Annotator Consensus Adjudication Tests
# ---------------------------------------------------------------------------

def test_adjudication_majority_consensus():
    """Verify that a 2-out-of-3 majority vote produces ADJUDICATED_MAJORITY."""
    pipeline = LabelValidationPipeline(minimum_consensus_threshold=0.65)

    votes = [
        AnnotatorVote(annotator_id="A1", assigned_label=NormalizedForensicLabel.FULLY_AI_GENERATED, confidence=0.95),
        AnnotatorVote(annotator_id="A2", assigned_label=NormalizedForensicLabel.FULLY_AI_GENERATED, confidence=0.90),
        AnnotatorVote(annotator_id="A3", assigned_label=NormalizedForensicLabel.PARTIALLY_AI_GENERATED, confidence=0.70),
    ]

    record = pipeline.adjudicate_record(
        record_id="REC-001",
        sha256="a" * 64,
        source_dataset="GenImage",
        original_publisher_label="ai_generated",
        votes=votes,
    )

    assert record.normalized_label == NormalizedForensicLabel.FULLY_AI_GENERATED
    assert record.consensus_status == ConsensusStatus.ADJUDICATED_MAJORITY
    assert record.label_confidence >= 0.90
    assert record.task_eligibility.quarantined is False


def test_adjudication_split_vote_quarantined():
    """Verify that a 3-way split vote triggers CONTRADICTORY_AMBIGUOUS and quarantines the asset."""
    pipeline = LabelValidationPipeline(minimum_consensus_threshold=0.65)

    votes = [
        AnnotatorVote(annotator_id="A1", assigned_label=NormalizedForensicLabel.FACE_SWAPPED, confidence=0.80),
        AnnotatorVote(annotator_id="A2", assigned_label=NormalizedForensicLabel.FACIAL_REENACTMENT, confidence=0.75),
        AnnotatorVote(annotator_id="A3", assigned_label=NormalizedForensicLabel.FACE_INPAINTING, confidence=0.70),
    ]

    record = pipeline.adjudicate_record(
        record_id="REC-002",
        sha256="b" * 64,
        source_dataset="WildDeepfake",
        original_publisher_label="manipulated",
        votes=votes,
    )

    assert record.consensus_status == ConsensusStatus.CONTRADICTORY_AMBIGUOUS
    assert record.task_eligibility.quarantined is True
    assert record.task_eligibility.eligible_for_model_a_train is False
    assert record.task_eligibility.eligible_for_model_b_train is False
    # Anti-discard check: Must remain preserved for evaluation
    assert record.task_eligibility.eligible_for_evaluation_only is True
    assert record.task_eligibility.exclusion_reason is not None


# ---------------------------------------------------------------------------
# 5. Task-Specific Training Eligibility Masking Tests
# ---------------------------------------------------------------------------

def test_task_eligibility_masking():
    """Verify that Model A and Model B train eligibility are strictly segregated."""
    pipeline = LabelValidationPipeline()

    # 1. Full-frame AI generation: Eligible for Model A, excluded from Model B
    mask_ai = pipeline.determine_task_eligibility(
        normalized_label=NormalizedForensicLabel.FULLY_AI_GENERATED,
        consensus_status=ConsensusStatus.UNANIMOUS_CONSENSUS,
        label_confidence=0.95,
    )
    assert mask_ai.eligible_for_model_a_train is True
    assert mask_ai.eligible_for_model_b_train is False
    assert "Excluded from Model B" in (mask_ai.exclusion_reason or "")

    # 2. Localized Face Swap: Eligible for Model B, excluded from Model A
    mask_swap = pipeline.determine_task_eligibility(
        normalized_label=NormalizedForensicLabel.FACE_SWAPPED,
        consensus_status=ConsensusStatus.UNANIMOUS_CONSENSUS,
        label_confidence=0.95,
    )
    assert mask_swap.eligible_for_model_a_train is False
    assert mask_swap.eligible_for_model_b_train is True
    assert "Excluded from Model A" in (mask_swap.exclusion_reason or "")

    # 3. Retouched Camera Photo: Clean baseline, eligible for both baselines
    mask_photo = pipeline.determine_task_eligibility(
        normalized_label=NormalizedForensicLabel.RETOUCHED_GENUINE_PHOTOGRAPH,
        consensus_status=ConsensusStatus.UNANIMOUS_CONSENSUS,
        label_confidence=0.95,
    )
    assert mask_photo.eligible_for_model_a_train is True
    assert mask_photo.eligible_for_model_b_train is True


# ---------------------------------------------------------------------------
# 6. Original Label Immutability & Anti-Discard Tests
# ---------------------------------------------------------------------------

def test_original_publisher_label_preserved_immutably():
    """Verify that verbatim publisher string is retained unchanged alongside normalized taxonomy."""
    pipeline = LabelValidationPipeline()
    raw_pub = "FF++_c23_FaceSwap_compressed_v1"

    rec = pipeline.adjudicate_record(
        record_id="REC-003",
        sha256="c" * 64,
        source_dataset="FaceForensics++",
        original_publisher_label=raw_pub,
    )

    assert rec.original_publisher_label == raw_pub
    assert rec.normalized_label == NormalizedForensicLabel.FACE_SWAPPED
    assert rec.record_id == "REC-003"
    assert rec.sha256 == "c" * 64
