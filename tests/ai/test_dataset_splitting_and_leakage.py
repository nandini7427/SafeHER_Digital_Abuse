"""Automated test suite for SafeHER Milestone 4.1 Dataset Splitting and Leakage Prevention.

Tests all required scenarios:
1. Identity / Subject Group isolation:
   - Same identity or actor grouping_id cannot leak between designated splits.
2. Original / Manipulated Lineage grouping:
   - Parent pristine images and all derived manipulated variants remain together in the same split.
3. Near-Duplicate cluster isolation:
   - Perceptual near-duplicate candidate pairs never cross partition boundaries.
4. Held-out generator isolation:
   - Records with held-out generators (e.g. DALL-E, Midjourney) are 100% held out from train/val.
5. Deterministic reproducibility:
   - Fixed random seed produces bit-for-bit identical split assignments.
6. Class balance preservation:
   - Stratified cluster allocation maintains balanced authentic vs manipulated ratios where feasible.
7. Impossible split configurations fail clearly:
   - Fewer than 3 clusters fails with explicit DatasetLeakageError.
   - Contradictory ratios or impossible partitions never silently leak records.
8. Comprehensive Leakage Audit Report:
   - LeakageDetector audits splits, detects cross-split identity/lineage leakage, and generates Markdown.
"""

import hashlib
from typing import List
import pytest

from ai_core.datasets.duplicates import NearDuplicateCandidate
from ai_core.datasets.leakage import (
    LeakageDetector,
    LeakageType,
)
from ai_core.datasets.manifest import (
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetSplit,
    DatasetTaskType,
    ManipulationCategory,
    RightsApprovalStatus,
    TransformationStep,
    UsageScope,
)
from ai_core.datasets.splitter import (
    DatasetLeakageError,
    GroupAwareDatasetSplitter,
    SplitPolicy,
)


def _make_dummy_hash(identifier: str) -> str:
    """Generate deterministic 64-character SHA-256 digest."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest().lower()


def _create_sample_record(
    record_id: str,
    label: DatasetImageLabel = DatasetImageLabel.CAMERA_PHOTOGRAPH,
    manipulation_category: ManipulationCategory = ManipulationCategory.NONE,
    grouping_id: str = None,
    original_image_ref: str = None,
    parent_image_ref: str = None,
    generator_name: str = None,
    source_dataset: str = "Test-Dataset",
) -> DatasetImageRecord:
    """Helper to build test records with explicit grouping and lineage links."""
    return DatasetImageRecord(
        record_id=record_id,
        image_path_or_ref=f"images/{record_id}.jpg",
        sha256=_make_dummy_hash(record_id),
        label=label,
        manipulation_category=manipulation_category,
        grouping_id=grouping_id,
        original_image_ref=original_image_ref,
        parent_image_ref=parent_image_ref,
        generator_name=generator_name,
        source_dataset=source_dataset,
        source_creator="SafeHER Test Studio",
        original_license="Apache-2.0",
        split=DatasetSplit.UNASSIGNED,
    )


# ------------------------------------------------------------------------------
# 1. Identity / Subject Group Isolation
# ------------------------------------------------------------------------------
def test_identity_isolation_across_splits() -> None:
    """Verify that multiple frames/images of the same identity are never split across train/val/test."""
    records: List[DatasetImageRecord] = []

    # Create 10 distinct subject identities, with 3 images each (30 images total)
    for i in range(1, 11):
        subj_id = f"SUBJ-{i:03d}"
        for j in range(1, 4):
            records.append(
                _create_sample_record(
                    record_id=f"REC-{subj_id}-IMG-{j}",
                    grouping_id=subj_id,
                    label=(
                        DatasetImageLabel.CAMERA_PHOTOGRAPH
                        if j == 1
                        else DatasetImageLabel.FACE_SWAP
                    ),
                    manipulation_category=(
                        ManipulationCategory.NONE
                        if j == 1
                        else ManipulationCategory.DEEPFAKE_FACE_SWAP
                    ),
                )
            )

    splitter = GroupAwareDatasetSplitter(
        policy=SplitPolicy(
            seed=42,
            train_ratio=0.70,
            val_ratio=0.15,
            internal_test_ratio=0.15,
            group_by_identity=True,
        )
    )

    result = splitter.split_records(records)
    detector = LeakageDetector()
    report = detector.audit_splits(result.all_partitioned_records)

    assert report.is_leakage_free is True
    assert report.total_violations_found == 0

    # Explicit check: Verify each identity exists in EXACTLY ONE split
    subj_to_splits = {}
    for r in result.all_partitioned_records:
        if r.grouping_id:
            subj_to_splits.setdefault(r.grouping_id, set()).add(r.split)

    for subj, splits in subj_to_splits.items():
        assert len(splits) == 1, f"Identity {subj} leaked across multiple splits: {splits}"


# ------------------------------------------------------------------------------
# 2. Original / Manipulated Lineage Grouping
# ------------------------------------------------------------------------------
def test_lineage_derivative_grouping() -> None:
    """Verify that original authentic images and all derived variants remain in the same split."""
    records: List[DatasetImageRecord] = []

    # 6 independent scene roots, each with 1 authentic parent + 2 synthetic derivatives
    for i in range(1, 7):
        root_id = f"ROOT-SCENE-{i:03d}"
        parent_id = f"PARENT-IMG-{i:03d}"
        # Authentic root
        records.append(
            _create_sample_record(
                record_id=parent_id,
                label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
                manipulation_category=ManipulationCategory.NONE,
                original_image_ref=root_id,
            )
        )
        # Derivative 1: ELA synthetic edit
        records.append(
            _create_sample_record(
                record_id=f"CHILD-ELA-{i:03d}",
                label=DatasetImageLabel.AI_GENERATED_IMAGE,
                manipulation_category=ManipulationCategory.SYNTHETIC_EDIT,
                original_image_ref=root_id,
                parent_image_ref=parent_id,
                generator_name="SafeHER-ELA-Shader",
            )
        )
        # Derivative 2: Spliced composite
        records.append(
            _create_sample_record(
                record_id=f"CHILD-SPLICE-{i:03d}",
                label=DatasetImageLabel.IMAGE_COMPOSITE,
                manipulation_category=ManipulationCategory.SPLICED_COMPOSITE,
                original_image_ref=root_id,
                parent_image_ref=parent_id,
                generator_name="SafeHER-Splicer",
            )
        )

    splitter = GroupAwareDatasetSplitter(
        policy=SplitPolicy(
            seed=101,
            train_ratio=0.60,
            val_ratio=0.20,
            internal_test_ratio=0.20,
            group_by_lineage=True,
        )
    )

    result = splitter.split_records(records)
    detector = LeakageDetector()
    report = detector.audit_splits(result.all_partitioned_records)

    assert report.is_leakage_free is True
    assert report.total_violations_found == 0

    # Verify root lineages never span multiple splits
    root_to_splits = {}
    for r in result.all_partitioned_records:
        root_to_splits.setdefault(r.original_image_ref, set()).add(r.split)

    for root_ref, splits in root_to_splits.items():
        assert len(splits) == 1, f"Root lineage {root_ref} leaked across splits: {splits}"


# ------------------------------------------------------------------------------
# 3. Near-Duplicate Cluster Isolation
# ------------------------------------------------------------------------------
def test_near_duplicate_cluster_isolation() -> None:
    """Verify that near-duplicate candidates (e.g. recompressions) are kept in the same split."""
    records = [
        _create_sample_record(f"REC-{i:02d}", label=DatasetImageLabel.CAMERA_PHOTOGRAPH)
        for i in range(1, 11)
    ]

    # Assume REC-01 and REC-02 are near duplicates (e.g. different recompression)
    near_dup_pair = NearDuplicateCandidate(
        record_id_a="REC-01",
        record_id_b="REC-02",
        hash_a="0000111100001111",
        hash_b="0000111100001110",
        hamming_distance=1,
        similarity_score=0.9844,
        category="HIGH_CONFIDENCE_RECOMPRESSION_OR_FORMAT_CONVERSION",
        requires_human_review=True,
        notes="Recompression pair",
    )

    splitter = GroupAwareDatasetSplitter(
        policy=SplitPolicy(
            seed=7,
            train_ratio=0.60,
            val_ratio=0.20,
            internal_test_ratio=0.20,
            group_by_near_duplicates=True,
        )
    )

    result = splitter.split_records(records, near_duplicate_pairs=[near_dup_pair])
    detector = LeakageDetector()
    report = detector.audit_splits(
        result.all_partitioned_records,
        near_duplicate_pairs=[near_dup_pair],
    )

    assert report.is_leakage_free is True

    # REC-01 and REC-02 MUST have identical split assignment
    rec_splits = {r.record_id: r.split for r in result.all_partitioned_records}
    assert rec_splits["REC-01"] == rec_splits["REC-02"]


# ------------------------------------------------------------------------------
# 4. Held-Out Generator Isolation
# ------------------------------------------------------------------------------
def test_held_out_generator_isolation() -> None:
    """Verify that records generated by held-out architectures are 100% quarantined to test."""
    records = []

    # In-domain generators (Stable Diffusion, GAN)
    for i in range(1, 10):
        records.append(
            _create_sample_record(
                f"REC-SD-{i}",
                label=DatasetImageLabel.AI_GENERATED_IMAGE,
                generator_name="StableDiffusion-v1.5",
            )
        )
    # Authentic camera
    for i in range(1, 10):
        records.append(
            _create_sample_record(
                f"REC-AUTH-{i}",
                label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            )
        )
    # Held-out generator (Midjourney-v6)
    for i in range(1, 6):
        records.append(
            _create_sample_record(
                f"REC-MJ-{i}",
                label=DatasetImageLabel.AI_GENERATED_IMAGE,
                generator_name="Midjourney-v6",
            )
        )

    splitter = GroupAwareDatasetSplitter(
        policy=SplitPolicy(
            seed=42,
            held_out_generators={"Midjourney-v6"},
        )
    )

    result = splitter.split_records(records)
    detector = LeakageDetector(held_out_generators={"Midjourney-v6"})
    report = detector.audit_splits(result.all_partitioned_records)

    assert report.is_leakage_free is True

    # Midjourney records MUST be assigned to HELD_OUT_GENERATOR_TEST
    for r in result.all_partitioned_records:
        if r.generator_name == "Midjourney-v6":
            assert r.split == DatasetSplit.HELD_OUT_GENERATOR_TEST
        else:
            assert r.split != DatasetSplit.HELD_OUT_GENERATOR_TEST


# ------------------------------------------------------------------------------
# 5. Deterministic Reproducibility
# ------------------------------------------------------------------------------
def test_deterministic_reproducibility_for_fixed_seed() -> None:
    """Verify that a fixed seed produces bit-for-bit identical partition assignments and hashes."""
    records = [
        _create_sample_record(
            f"REC-{i:03d}",
            grouping_id=f"GRP-{i % 5}",
            label=(
                DatasetImageLabel.CAMERA_PHOTOGRAPH
                if i % 2 == 0
                else DatasetImageLabel.AI_GENERATED_IMAGE
            ),
        )
        for i in range(1, 21)
    ]

    policy_42 = SplitPolicy(seed=42)
    splitter_1 = GroupAwareDatasetSplitter(policy_42)
    splitter_2 = GroupAwareDatasetSplitter(policy_42)

    result_1 = splitter_1.split_records(records)
    result_2 = splitter_2.split_records(records)

    assert result_1.split_manifest_hash == result_2.split_manifest_hash
    for r1, r2 in zip(result_1.all_partitioned_records, result_2.all_partitioned_records):
        assert r1.record_id == r2.record_id
        assert r1.split == r2.split


# ------------------------------------------------------------------------------
# 6. Class Balance Preservation
# ------------------------------------------------------------------------------
def test_class_balance_stratification() -> None:
    """Verify that stratified cluster allocation maintains authentic vs manipulated balance."""
    records = []
    # 6 authentic clusters
    for i in range(1, 7):
        records.append(
            _create_sample_record(
                f"AUTH-CLUSTER-{i}",
                grouping_id=f"AUTH-GRP-{i}",
                label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            )
        )
    # 6 manipulated clusters
    for i in range(1, 7):
        records.append(
            _create_sample_record(
                f"MANIP-CLUSTER-{i}",
                grouping_id=f"MANIP-GRP-{i}",
                label=DatasetImageLabel.AI_GENERATED_IMAGE,
                manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
            )
        )

    splitter = GroupAwareDatasetSplitter(
        policy=SplitPolicy(
            seed=42,
            train_ratio=0.60,
            val_ratio=0.20,
            internal_test_ratio=0.20,
            stratify_by_label=True,
        )
    )

    result = splitter.split_records(records)
    train_recs = result.records_by_split[DatasetSplit.TRAIN]
    val_recs = result.records_by_split[DatasetSplit.VAL]
    test_recs = result.records_by_split[DatasetSplit.INTERNAL_TEST]

    # Verify every split contains at least one authentic and one manipulated asset
    assert any(r.label == DatasetImageLabel.CAMERA_PHOTOGRAPH for r in train_recs)
    assert any(r.label == DatasetImageLabel.AI_GENERATED_IMAGE for r in train_recs)
    assert any(r.label == DatasetImageLabel.CAMERA_PHOTOGRAPH for r in val_recs)
    assert any(r.label == DatasetImageLabel.AI_GENERATED_IMAGE for r in val_recs)
    assert any(r.label == DatasetImageLabel.CAMERA_PHOTOGRAPH for r in test_recs)
    assert any(r.label == DatasetImageLabel.AI_GENERATED_IMAGE for r in test_recs)


# ------------------------------------------------------------------------------
# 7. Impossible Split Configurations Fail Clearly
# ------------------------------------------------------------------------------
def test_impossible_split_configuration_fails_clearly() -> None:
    """Verify that insufficient clusters or impossible constraints fail explicitly without silent leakage."""
    # Only 2 clusters available for train/val/test (need at least 3)
    rec1 = _create_sample_record("REC-1", grouping_id="GRP-1")
    rec2 = _create_sample_record("REC-2", grouping_id="GRP-1")  # Same group as REC-1
    rec3 = _create_sample_record("REC-3", grouping_id="GRP-2")

    splitter = GroupAwareDatasetSplitter()

    with pytest.raises(DatasetLeakageError, match="Impossible split configuration"):
        splitter.split_records([rec1, rec2, rec3])


# ------------------------------------------------------------------------------
# 8. Comprehensive Leakage Audit Report
# ------------------------------------------------------------------------------
def test_leakage_detector_identifies_cross_split_contamination() -> None:
    """Verify LeakageDetector catches intentional identity and lineage leakage across splits."""
    rec_train = _create_sample_record("REC-TR-01", grouping_id="ACTOR-X")
    rec_train = rec_train.model_copy(update={"split": DatasetSplit.TRAIN})

    # Contaminated record: Same actor in TEST!
    rec_test = _create_sample_record("REC-TE-01", grouping_id="ACTOR-X")
    rec_test = rec_test.model_copy(update={"split": DatasetSplit.TEST})

    detector = LeakageDetector()
    report = detector.audit_splits([rec_train, rec_test])

    assert report.is_leakage_free is False
    assert report.total_violations_found == 1
    assert report.violations[0].leakage_type == LeakageType.IDENTITY_LEAKAGE
    assert "ACTOR-X" in report.violations[0].details

    # Markdown format verification
    md = report.to_markdown()
    assert "# SafeHER Dataset Leakage Audit Report" in md
    assert "LEAKAGE DETECTED (FAILED)" in md
    assert "IDENTITY_LEAKAGE" in md
