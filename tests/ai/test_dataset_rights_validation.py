"""Automated test suite for SafeHER Milestone 4.1 Dataset Registry and Rights Validation.

Tests all required scenarios:
1. Valid research record.
2. Valid commercial record with required rights evidence.
3. Missing license.
4. Non-commercial dataset rejected for commercial training.
5. Missing image hash.
6. Duplicate image hash and conflicting provenance.
7. Missing attribution metadata.
8. Missing actor consent reference.
9. Unverified generator rights.
10. Mixed-license aggregate dataset (umbrella declaration cannot override upstream restrictions).
11. Research-only record attempting to enter production training.
12. Unseen generator metadata.

Plus safety guarantees:
- PII rejection in consent references.
- Initial registry completeness and NOT_APPROVED_FOR_PRODUCTION marking.
- Manifest serialization and cryptographic hash verification.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import pytest

from ai_core.datasets.manifest import (
    AttributionMetadata,
    CommercialUseStatus,
    ConsentStatus,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    GeneratorInfo,
    GroundTruthLabel,
    ManipulationCategory,
    RightsApprovalStatus,
    TargetUsageScope,
    TransformationStep,
    UsageScope,
    compute_sha256,
    validate_non_identifying_consent_reference,
)
from ai_core.datasets.registry import (
    DATASET_REGISTRY,
    CandidateCategory,
    create_candidate_sample_record,
    create_initial_registry_manifest,
    get_dataset_profile,
    list_all_candidates,
    list_commercial_candidates,
    list_research_only_candidates,
)
from ai_core.datasets.rights_validator import (
    RightsReviewStatus,
    RightsValidationEngine,
    ValidationSeverity,
)


@pytest.fixture
def validator() -> RightsValidationEngine:
    """Fixture providing clean instance of the rights validation engine."""
    return RightsValidationEngine()


def _make_dummy_hash(identifier: str) -> str:
    """Generate a valid deterministic 64-character SHA-256 digest."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------
# Test 1: Valid research record
# ------------------------------------------------------------------------------
def test_valid_research_record(validator: RightsValidationEngine) -> None:
    """Verify that a compliant research record passes validation for research/evaluation target."""
    record = DatasetImageRecord(
        record_id="SH-RES-001",
        image_path_or_ref="faceforensics://c23/001_002.mp4#frame_10",
        sha256=_make_dummy_hash("ffpp_frame_10"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DEEPFAKE_FACE_SWAP,
        source_dataset="FaceForensics++",
        source_creator="Technical University of Munich",
        original_license="FaceForensics-Academic-Agreement",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        human_subjects_present=True,
        consent_reference="CR-RESEARCH-ACADEMIC-001",
        consent_status=ConsentStatus.RESEARCH_ONLY,
        attribution_requirements="FaceForensics++ (Roessler et al., ICCV 2019)",
        split=DatasetSplit.TEST,
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.RESEARCH,
    )

    assert report.is_valid is True
    assert report.status == RightsReviewStatus.APPROVED_FOR_TARGET
    assert report.total_records == 1
    assert report.passed_records == 1
    assert report.failed_records == 0
    assert len(report.violations) == 0


# ------------------------------------------------------------------------------
# Test 2: Valid commercial record with required rights evidence
# ------------------------------------------------------------------------------
def test_valid_commercial_record_with_required_rights_evidence(validator: RightsValidationEngine) -> None:
    """Verify that a commercial record with verified consent, license, and generator passes production training."""
    record = DatasetImageRecord(
        record_id="SH-COMM-001",
        image_path_or_ref="safeher://internal/studio/portrait_001.jpg",
        sha256=_make_dummy_hash("safeher_studio_001"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="First-Party Consent-Based Photographs",
        source_creator="SafeHER Internal Evidence Studio",
        original_license="SafeHER-First-Party-Commercial",
        license_url="safeher://agreements/commercial_likeness_release.pdf",
        allowed_usage=[
            UsageScope.RESEARCH,
            UsageScope.COMMERCIAL_TRAINING,
            UsageScope.EVALUATION,
            UsageScope.REDISTRIBUTION,
        ],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        human_subjects_present=True,
        consent_reference="CR-SAFEHER-2026-F001",
        consent_status=ConsentStatus.AUTHORIZED,
        attribution_requirements="SafeHER Internal Studio",
        split=DatasetSplit.TRAIN,
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.PRODUCTION_TRAINING,
    )

    assert report.is_valid is True
    assert report.status == RightsReviewStatus.APPROVED_FOR_TARGET
    assert report.total_records == 1
    assert report.passed_records == 1
    assert report.failed_records == 0
    assert len(report.violations) == 0


# ------------------------------------------------------------------------------
# Test 3: Missing license
# ------------------------------------------------------------------------------
def test_missing_license(validator: RightsValidationEngine) -> None:
    """Verify that an unknown or missing license is rejected for commercial training."""
    record = DatasetImageRecord(
        record_id="SH-NOLIC-001",
        image_path_or_ref="unknown://dataset/image01.jpg",
        sha256=_make_dummy_hash("unknown_lic_image"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Scraped-Web-Dump",
        source_creator="Anonymous",
        original_license="Unknown",
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.COMMERCIAL_TRAINING,
    )

    assert report.is_valid is False
    assert report.status == RightsReviewStatus.REJECTED
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_UNKNOWN_LICENSE" in rule_codes


# ------------------------------------------------------------------------------
# Test 4: Non-commercial dataset rejected for commercial training
# ------------------------------------------------------------------------------
def test_non_commercial_dataset_rejected_for_commercial_training(validator: RightsValidationEngine) -> None:
    """Verify that non-commercial datasets are rejected from commercial training and flagged research-only."""
    record = DatasetImageRecord(
        record_id="SH-NC-001",
        image_path_or_ref="cc_nc://dataset/img_01.jpg",
        sha256=_make_dummy_hash("cc_nc_sample"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Non-Commercial-Photo-Collection",
        source_creator="Photographer-A",
        original_license="CC-BY-NC-4.0",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.COMMERCIAL_TRAINING],  # Erroneous commercial claim
        rights_status=RightsApprovalStatus.NOT_APPROVED_FOR_PRODUCTION,
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.COMMERCIAL_TRAINING,
    )

    assert report.is_valid is False
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_NON_COMMERCIAL_RESTRICTION" in rule_codes


# ------------------------------------------------------------------------------
# Test 5: Missing image hash
# ------------------------------------------------------------------------------
def test_missing_image_hash(validator: RightsValidationEngine) -> None:
    """Verify that missing or malformed SHA-256 hashes are caught and rejected."""
    # Pydantic validation rejects malformed SHA-256 at model instantiation
    with pytest.raises(ValueError, match="Invalid SHA-256 digest format"):
        DatasetImageRecord(
            record_id="SH-BADHASH-001",
            image_path_or_ref="local://path/img.jpg",
            sha256="not_a_valid_sha256_hash",
            label=GroundTruthLabel.AUTHENTIC,
            manipulation_category=ManipulationCategory.NONE,
            source_dataset="TestDS",
            source_creator="Creator",
            original_license="Apache-2.0",
        )

    with pytest.raises(ValueError, match="SHA-256 hash must not be empty"):
        DatasetImageRecord(
            record_id="SH-BADHASH-002",
            image_path_or_ref="local://path/img.jpg",
            sha256="",
            label=GroundTruthLabel.AUTHENTIC,
            manipulation_category=ManipulationCategory.NONE,
            source_dataset="TestDS",
            source_creator="Creator",
            original_license="Apache-2.0",
        )


# ------------------------------------------------------------------------------
# Test 6: Duplicate image hash
# ------------------------------------------------------------------------------
def test_duplicate_image_hash(validator: RightsValidationEngine) -> None:
    """Verify detection of duplicate image hashes and contradictory provenance."""
    shared_hash = _make_dummy_hash("identical_image_bytes")

    rec1 = DatasetImageRecord(
        record_id="SH-DUP-001",
        image_path_or_ref="ds_a://image.jpg",
        sha256=shared_hash,
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Dataset-Alpha",
        source_creator="Author-1",
        original_license="Apache-2.0",
    )

    rec2 = DatasetImageRecord(
        record_id="SH-DUP-002",
        image_path_or_ref="ds_b://cloned_image.jpg",
        sha256=shared_hash,
        label=GroundTruthLabel.MANIPULATED,  # Contradictory label!
        manipulation_category=ManipulationCategory.SPLICED_COMPOSITE,
        source_dataset="Dataset-Beta",      # Contradictory source!
        source_creator="Author-2",
        original_license="MIT",
    )

    manifest = DatasetManifest(
        manifest_id="SH-MAN-DUP-TEST",
        dataset_name="Duplicate Test Manifest",
        target_use=TargetUsageScope.EVALUATION,
        records=[rec1, rec2],
    )

    report = validator.validate_manifest(manifest)

    assert report.is_valid is False
    assert shared_hash in report.duplicate_hashes_found
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_DUPLICATE_IMAGE_HASH" in rule_codes
    assert "ERR_CONFLICTING_PROVENANCE" in rule_codes


# ------------------------------------------------------------------------------
# Test 7: Missing attribution
# ------------------------------------------------------------------------------
def test_missing_attribution(validator: RightsValidationEngine) -> None:
    """Verify that licenses mandating attribution reject records lacking attribution notices."""
    record = DatasetImageRecord(
        record_id="SH-ATTRIB-001",
        image_path_or_ref="openimages://image_001.jpg",
        sha256=_make_dummy_hash("openimages_001"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Open Images",
        source_creator="Flickr Contributor 123",  # Creator present
        original_license="CC-BY-4.0",  # Requires attribution
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        attribution_requirements=None,  # Missing attribution notice text
        attribution=None,
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.PRODUCTION_TRAINING,
    )

    assert report.is_valid is False
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_MISSING_ATTRIBUTION" in rule_codes


# ------------------------------------------------------------------------------
# Test 8: Missing actor consent reference
# ------------------------------------------------------------------------------
def test_missing_actor_consent_reference(validator: RightsValidationEngine) -> None:
    """Verify that face-based commercial training data requires explicit authorized consent."""
    record = DatasetImageRecord(
        record_id="SH-NOCONSENT-001",
        image_path_or_ref="commercial://faces/actor_01.jpg",
        sha256=_make_dummy_hash("face_actor_01"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Commercial-Portrait-Set",
        source_creator="Studio-B",
        original_license="Apache-2.0",
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        human_subjects_present=True,
        consent_reference=None,  # MISSING CONSENT REFERENCE
        consent_status=ConsentStatus.UNVERIFIED,
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.COMMERCIAL_TRAINING,
    )

    assert report.is_valid is False
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_MISSING_ACTOR_CONSENT" in rule_codes


# ------------------------------------------------------------------------------
# Test 9: Unverified generator rights
# ------------------------------------------------------------------------------
def test_unverified_generator_rights(validator: RightsValidationEngine) -> None:
    """Verify that synthetic images from unverified generators are rejected for commercial training."""
    record = DatasetImageRecord(
        record_id="SH-UNVERIF-GEN-001",
        image_path_or_ref="synthetic://sdxl/sample_01.jpg",
        sha256=_make_dummy_hash("sdxl_sample_01"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
        source_dataset="Synthetic-SDXL-Collection",
        source_creator="Online Platform",
        original_license="Apache-2.0",
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        generator_name="StableDiffusion-v1.4",
        generator_info=GeneratorInfo(
            name="StableDiffusion-v1.4",
            commercial_use_permitted=False,  # NOT PERMITTED
        ),
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.COMMERCIAL_TRAINING,
    )

    assert report.is_valid is False
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_UNVERIFIED_GENERATOR_RIGHTS" in rule_codes


# ------------------------------------------------------------------------------
# Test 10: Mixed-license aggregate dataset
# ------------------------------------------------------------------------------
def test_mixed_license_aggregate_dataset(validator: RightsValidationEngine) -> None:
    """Verify that an aggregate general license cannot override individual upstream restrictions."""
    rec_commercial = DatasetImageRecord(
        record_id="SH-AGG-001",
        image_path_or_ref="openimages://item_cc0.jpg",
        sha256=_make_dummy_hash("agg_cc0"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Open Images",
        source_creator="Flickr Contributor 1",
        original_license="CC0-1.0",
        original_source_ref="https://flickr.com/photos/123/cc0",
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        is_aggregated=True,
    )

    rec_restricted_nc = DatasetImageRecord(
        record_id="SH-AGG-002",
        image_path_or_ref="openimages://item_nc.jpg",
        sha256=_make_dummy_hash("agg_nc"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Open Images",
        source_creator="Flickr Contributor 2",
        original_license="CC-BY-NC-4.0",  # UPSTREAM RESTRICTION!
        original_source_ref="https://flickr.com/photos/456/nc",
        allowed_usage=[UsageScope.RESEARCH],
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        is_aggregated=True,
    )

    manifest = DatasetManifest(
        manifest_id="SH-MAN-AGG-MIXED",
        dataset_name="Open Images Subset",
        target_use=TargetUsageScope.COMMERCIAL_TRAINING,
        is_aggregated=True,
        general_license="CC-BY-4.0",  # Broad umbrella claim
        records=[rec_commercial, rec_restricted_nc],
    )

    report = validator.validate_manifest(manifest)

    assert report.is_valid is False
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_AGGREGATE_UPSTREAM_RESTRICTION_CONFLICT" in rule_codes


# ------------------------------------------------------------------------------
# Test 11: Research-only record attempting to enter production training
# ------------------------------------------------------------------------------
def test_research_only_record_attempting_to_enter_production_training(
    validator: RightsValidationEngine,
) -> None:
    """Verify strict isolation preventing research-only images from entering production manifests."""
    record = DatasetImageRecord(
        record_id="SH-RO-PROD-001",
        image_path_or_ref="deeperforensics://sample_01.jpg",
        sha256=_make_dummy_hash("df_sample_01"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DEEPFAKE_FACE_SWAP,
        source_dataset="DeeperForensics-1.0",
        source_creator="NTU / SenseTime",
        original_license="DeeperForensics-Academic-License",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        human_subjects_present=True,
        consent_reference="CR-RESEARCH-ACTOR-001",
        consent_status=ConsentStatus.RESEARCH_ONLY,
        attribution_requirements="DeeperForensics-1.0 (Jiang et al., CVPR 2020)",
        split=DatasetSplit.TRAIN,
    )

    manifest = DatasetManifest(
        manifest_id="SH-MAN-PRODUCTION-LEAK-TEST",
        dataset_name="Production Training Candidates",
        target_use=TargetUsageScope.PRODUCTION_TRAINING,
        records=[record],
    )

    report = validator.validate_manifest(manifest)

    assert report.is_valid is False
    assert report.status == RightsReviewStatus.REJECTED
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_RESEARCH_ONLY_IN_PRODUCTION" in rule_codes


# ------------------------------------------------------------------------------
# Test 12: Unseen generator metadata
# ------------------------------------------------------------------------------
def test_unseen_generator_metadata(validator: RightsValidationEngine) -> None:
    """Verify that an unseen generator without registered terms triggers an actionable requirement."""
    record = DatasetImageRecord(
        record_id="SH-UNSEEN-001",
        image_path_or_ref="synthetic://unseen_tool/sample_01.jpg",
        sha256=_make_dummy_hash("unseen_tool_sample"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
        source_dataset="Experimental-Synthetic-Bench",
        source_creator="Lab-X",
        original_license="Apache-2.0",
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        generator_name="Wild-Diffusion-Model-9000",  # Unregistered generator
        generator_version="0.1.0-alpha",
        generator_info=None,  # No registered terms
    )

    report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.COMMERCIAL_TRAINING,
    )

    assert report.is_valid is False
    rule_codes = [v.rule_code for v in report.violations]
    assert "ERR_UNSEEN_GENERATOR_METADATA" in rule_codes


# ------------------------------------------------------------------------------
# Additional Tests: Safety, Privacy & Registry Completeness
# ------------------------------------------------------------------------------
def test_consent_reference_pii_rejection() -> None:
    """Ensure that storing PII (emails, phone numbers, SSNs) in consent_reference is blocked."""
    with pytest.raises(ValueError, match="Storing sensitive PII .* is strictly prohibited"):
        DatasetImageRecord(
            record_id="SH-PII-001",
            image_path_or_ref="local://path.jpg",
            sha256=_make_dummy_hash("pii_test"),
            label=GroundTruthLabel.AUTHENTIC,
            manipulation_category=ManipulationCategory.NONE,
            source_dataset="Test",
            source_creator="Test",
            original_license="Apache-2.0",
            consent_reference="john.doe@example.com",  # EMAIL PII
        )

    with pytest.raises(ValueError, match="Storing sensitive PII .* is strictly prohibited"):
        DatasetImageRecord(
            record_id="SH-PII-002",
            image_path_or_ref="local://path.jpg",
            sha256=_make_dummy_hash("pii_test_phone"),
            label=GroundTruthLabel.AUTHENTIC,
            manipulation_category=ManipulationCategory.NONE,
            source_dataset="Test",
            source_creator="Test",
            original_license="Apache-2.0",
            consent_reference="Phone: 555-123-4567",  # PHONE PII
        )


def test_initial_registry_completeness_and_statuses() -> None:
    """Verify that all 9 candidate datasets exist and all initial samples default to NOT_APPROVED_FOR_PRODUCTION."""
    candidates = list_all_candidates()
    assert len(candidates) == 9

    # Research-only or restricted
    research_ids = {c.dataset_id for c in list_research_only_candidates()}
    expected_research = {
        "genimage",
        "faceforensics_pp",
        "deeperforensics_1_0",
        "dfdc",
        "eddc_2026",
    }
    assert research_ids == expected_research

    # Commercial candidates requiring provenance review
    commercial_ids = {c.dataset_id for c in list_commercial_candidates()}
    expected_commercial = {
        "real_vs_ai",
        "open_images",
        "first_party_consent_photos",
        "commercial_synthetic_edits",
    }
    assert commercial_ids == expected_commercial

    # Master manifest verification
    master_manifest = create_initial_registry_manifest()
    assert len(master_manifest.records) == 9
    for rec in master_manifest.records:
        assert rec.rights_status == RightsApprovalStatus.NOT_APPROVED_FOR_PRODUCTION


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    """Verify JSON persistence and roundtrip fidelity."""
    manifest = create_initial_registry_manifest()
    json_path = tmp_path / "manifest.json"
    manifest.save(json_path)

    loaded = DatasetManifest.load(json_path)
    assert loaded.manifest_id == manifest.manifest_id
    assert len(loaded.records) == len(manifest.records)
    for orig, res in zip(manifest.records, loaded.records):
        assert orig.record_id == res.record_id
        assert orig.sha256 == res.sha256


def test_sha256_computation_utility(tmp_path: Path) -> None:
    """Verify SHA-256 computation utility across bytes, strings, and files."""
    test_bytes = b"SafeHER Forensic Immutability Verification 2026"
    expected = hashlib.sha256(test_bytes).hexdigest()

    # Bytes
    assert compute_sha256(test_bytes) == expected

    # String
    assert compute_sha256("SafeHER Forensic Immutability Verification 2026") == expected

    # File
    f = tmp_path / "sample.bin"
    f.write_bytes(test_bytes)
    assert compute_sha256(f) == expected


# ------------------------------------------------------------------------------
# Milestone 4.1 Required Licensing & Rights Verification Scenarios
# ------------------------------------------------------------------------------
def test_expired_review_record(validator: RightsValidationEngine) -> None:
    """Verify that records with EXPIRED_OR_REQUIRES_REVIEW status block commercial training and warn research."""
    record = DatasetImageRecord(
        record_id="SH-EXP-001",
        image_path_or_ref="safeher://internal/studio/sample_expired.jpg",
        sha256=_make_dummy_hash("sample_expired_review"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="First-Party Consent-Based Photographs",
        source_creator="SafeHER Internal Studio",
        original_license="SafeHER-First-Party-Commercial",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.EXPIRED_OR_REQUIRES_REVIEW,
        human_subjects_present=True,
        consent_reference="CR-SAFEHER-2026-F099",
        consent_status=ConsentStatus.AUTHORIZED,
        attribution_requirements="SafeHER Internal Studio",
        split=DatasetSplit.TRAIN,
    )

    # 1. Commercial training target: MUST be BLOCKED
    comm_report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.COMMERCIAL_TRAINING,
    )
    assert comm_report.is_valid is False
    assert comm_report.status == RightsReviewStatus.REJECTED
    rule_codes = [v.rule_code for v in comm_report.violations]
    assert "ERR_EXPIRED_OR_REQUIRES_REVIEW" in rule_codes
    assert any(
        v.severity == ValidationSeverity.BLOCKER and v.rule_code == "ERR_EXPIRED_OR_REQUIRES_REVIEW"
        for v in comm_report.violations
    )

    # 2. Research target: Flags a WARNING requiring re-audit
    res_report = validator.validate_records(
        records=[record],
        target_usage=TargetUsageScope.RESEARCH,
    )
    assert res_report.is_valid is True  # Non-blocking for research
    res_rule_codes = [v.rule_code for v in res_report.violations]
    assert "ERR_EXPIRED_OR_REQUIRES_REVIEW" in res_rule_codes
    assert any(
        v.severity == ValidationSeverity.WARNING and v.rule_code == "ERR_EXPIRED_OR_REQUIRES_REVIEW"
        for v in res_report.violations
    )


def test_conflicting_terms_record(validator: RightsValidationEngine) -> None:
    """Verify that records with pending review or contradictory internal terms are blocked."""
    # Scenario A: PENDING_REVIEW rights status
    record_pending = DatasetImageRecord(
        record_id="SH-CONF-001",
        image_path_or_ref="dfdc://sample_pending.mp4#frame_1",
        sha256=_make_dummy_hash("pending_review_sample"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DEEPFAKE_FACE_SWAP,
        source_dataset="DFDC",
        source_creator="Meta / AWS / Partnership on AI",
        original_license="DFDC-Challenge-Agreement",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.PENDING_REVIEW,
        human_subjects_present=True,
        consent_reference="CR-RESEARCH-ACTOR-001",
        consent_status=ConsentStatus.AUTHORIZED,
        attribution_requirements="DFDC Dataset",
        split=DatasetSplit.TRAIN,
    )

    report_pending = validator.validate_records(
        records=[record_pending],
        target_usage=TargetUsageScope.PRODUCTION_TRAINING,
    )
    assert report_pending.is_valid is False
    assert "ERR_CONFLICTING_RESTRICTIONS_PENDING_REVIEW" in [v.rule_code for v in report_pending.violations]

    # Scenario B: Internal contradiction: commercial_use_status=PROHIBITED but claiming COMMERCIAL_TRAINING
    record_contradiction = DatasetImageRecord(
        record_id="SH-CONF-002",
        image_path_or_ref="internal://contradiction.jpg",
        sha256=_make_dummy_hash("contradiction_sample"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Contradictory Collection",
        source_creator="Vendor-X",
        original_license="Apache-2.0",
        commercial_use_status=CommercialUseStatus.PROHIBITED,  # CONTRADICTION!
        allowed_usage=[UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    report_contradiction = validator.validate_records(
        records=[record_contradiction],
        target_usage=TargetUsageScope.PRODUCTION_TRAINING,
    )
    assert report_contradiction.is_valid is False
    assert "ERR_CONFLICTING_RESTRICTIONS_PENDING_REVIEW" in [v.rule_code for v in report_contradiction.violations]


def test_missing_upstream_license_in_aggregate(validator: RightsValidationEngine) -> None:
    """Verify that aggregated records missing original source references or conflicting upstream licenses are blocked."""
    # 1. Aggregated record missing original_source_ref
    record_no_upstream_ref = DatasetImageRecord(
        record_id="SH-AGG-001",
        image_path_or_ref="openimages://v7/sample_001.jpg",
        sha256=_make_dummy_hash("openimages_sample_001"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Open Images",
        source_creator="Flickr Creator",
        original_license="CC-BY-4.0",
        is_aggregated=True,
        original_source_ref=None,  # Missing upstream reference!
        allowed_usage=[UsageScope.RESEARCH],
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        split=DatasetSplit.TEST,
    )

    report_agg = validator.validate_records(
        records=[record_no_upstream_ref],
        target_usage=TargetUsageScope.RESEARCH,
    )
    assert report_agg.is_valid is False
    assert "ERR_AGGREGATE_SOURCE_UNVERIFIED" in [v.rule_code for v in report_agg.violations]

    # 2. General commercial license cannot override upstream non-commercial restriction
    record_upstream_conflict = DatasetImageRecord(
        record_id="SH-AGG-002",
        image_path_or_ref="openimages://v7/sample_nc.jpg",
        sha256=_make_dummy_hash("openimages_sample_nc"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Open Images",
        source_creator="Flickr Photographer",
        original_license="CC-BY-NC-4.0",  # Upstream is NC!
        original_source_ref="https://flickr.com/photos/user/123",
        is_aggregated=True,
        allowed_usage=[UsageScope.RESEARCH],
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        split=DatasetSplit.TEST,
    )

    manifest_conflict = DatasetManifest(
        manifest_id="SH-MAN-AGG-CONFLICT",
        dataset_name="Open Images Aggregate",
        version="1.0.0",
        target_use=TargetUsageScope.PRODUCTION_TRAINING,
        is_aggregated=True,
        general_license="CC-BY-4.0",  # Declared commercial general license
        records=[record_upstream_conflict],
    )

    report_conflict = validator.validate_manifest(
        manifest=manifest_conflict,
        target_usage=TargetUsageScope.PRODUCTION_TRAINING,
    )
    assert report_conflict.is_valid is False
    rule_codes = [v.rule_code for v in report_conflict.violations]
    assert "ERR_AGGREGATE_UPSTREAM_RESTRICTION_CONFLICT" in rule_codes
    assert "ERR_NON_COMMERCIAL_RESTRICTION" in rule_codes


def test_mixed_rights_manifest(validator: RightsValidationEngine) -> None:
    """Verify validation behavior on a mixed manifest containing research, evaluation, pending, and commercial records."""
    rec_research = DatasetImageRecord(
        record_id="SH-MIX-001-GENIMAGE",
        image_path_or_ref="genimage://sd15/001.jpg",
        sha256=_make_dummy_hash("mix_genimage"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
        source_dataset="GenImage",
        source_creator="Zhu et al.",
        original_license="GenImage-Academic-License",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        attribution_requirements="GenImage",
        split=DatasetSplit.TEST,
    )

    rec_eval_only = DatasetImageRecord(
        record_id="SH-MIX-002-EDDC",
        image_path_or_ref="eddc://2026/test_002.jpg",
        sha256=_make_dummy_hash("mix_eddc"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
        source_dataset="Explainable Deepfake Detection Challenge 2026",
        source_creator="EDDC Consortium",
        original_license="EDDC-2026-Academic",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        rights_status=RightsApprovalStatus.APPROVED_FOR_EVALUATION_ONLY,
        attribution_requirements="EDDC 2026",
        split=DatasetSplit.TEST,
    )

    rec_pending = DatasetImageRecord(
        record_id="SH-MIX-003-DFDC",
        image_path_or_ref="dfdc://clip_003.mp4#frame_1",
        sha256=_make_dummy_hash("mix_dfdc"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.DEEPFAKE_FACE_SWAP,
        source_dataset="DFDC",
        source_creator="Meta / AWS",
        original_license="DFDC-Challenge-Agreement",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        rights_status=RightsApprovalStatus.PENDING_REVIEW,
        attribution_requirements="DFDC",
        split=DatasetSplit.TEST,
    )

    rec_commercial = DatasetImageRecord(
        record_id="SH-MIX-004-SYNTH",
        image_path_or_ref="safeher://synthetic/ela_perturbation_004.jpg",
        sha256=_make_dummy_hash("mix_commercial_synth"),
        label=GroundTruthLabel.MANIPULATED,
        manipulation_category=ManipulationCategory.SYNTHETIC_EDIT,
        source_dataset="Commercially Permitted Synthetic Images and Edits",
        source_creator="SafeHER Synthetic Forensics Engine",
        original_license="Apache-2.0",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.COMMERCIAL_TRAINING, UsageScope.EVALUATION],
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        generator_name="safeher-procedural-v1",
        generator_version="1.0",
        generator_info=GeneratorInfo(
            name="safeher-procedural-v1",
            version="1.0",
            commercial_use_permitted=True,
            terms_reference="https://www.apache.org/licenses/LICENSE-2.0",
        ),
        attribution_requirements="SafeHER Synthetic Forensics Generator (Apache-2.0)",
        split=DatasetSplit.TRAIN,
    )

    manifest = DatasetManifest(
        manifest_id="SH-MAN-MIXED-001",
        dataset_name="Mixed Rights Benchmark Manifest",
        version="1.0.0",
        target_use=TargetUsageScope.PRODUCTION_TRAINING,
        records=[rec_research, rec_eval_only, rec_pending, rec_commercial],
    )

    # 1. Evaluated for PRODUCTION_TRAINING: Must fail, passing only the 1 commercial record
    prod_report = validator.validate_manifest(manifest=manifest, target_usage=TargetUsageScope.PRODUCTION_TRAINING)
    assert prod_report.is_valid is False
    assert prod_report.status == RightsReviewStatus.REJECTED
    assert prod_report.total_records == 4
    assert prod_report.passed_records == 1  # Only rec_commercial
    assert prod_report.failed_records == 3  # research, eval_only, and pending failed

    # 2. Evaluated for RESEARCH: Research records are permitted
    res_report = validator.validate_manifest(manifest=manifest, target_usage=TargetUsageScope.RESEARCH)
    assert res_report.is_valid is True
    assert res_report.status in (RightsReviewStatus.APPROVED_FOR_TARGET, RightsReviewStatus.ACTION_REQUIRED_UNRESOLVED)
    assert res_report.passed_records == 4


def test_rights_status_immutability_and_no_automatic_upgrade(validator: RightsValidationEngine) -> None:
    """Verify Rule 9: Validation engine NEVER mutates or automatically upgrades records' rights_status."""
    original_statuses = [
        RightsApprovalStatus.UNVERIFIED,
        RightsApprovalStatus.RESEARCH_ONLY,
        RightsApprovalStatus.PENDING_REVIEW,
        RightsApprovalStatus.APPROVED_FOR_EVALUATION_ONLY,
    ]

    records = [
        DatasetImageRecord(
            record_id=f"SH-IMMUTABLE-{i:03d}",
            image_path_or_ref=f"safeher://test/img_{i}.jpg",
            sha256=_make_dummy_hash(f"immutable_{i}"),
            label=GroundTruthLabel.AUTHENTIC,
            manipulation_category=ManipulationCategory.NONE,
            source_dataset="Test Immutable",
            source_creator="Test Creator",
            original_license="Apache-2.0",
            allowed_usage=[UsageScope.RESEARCH],
            rights_status=status,
            split=DatasetSplit.TEST,
        )
        for i, status in enumerate(original_statuses)
    ]

    # Validate against production training
    validator.validate_records(records=records, target_usage=TargetUsageScope.PRODUCTION_TRAINING)
    for rec, orig in zip(records, original_statuses):
        assert rec.rights_status == orig, f"Record '{rec.record_id}' was mutated from {orig} to {rec.rights_status}"

    # Validate against research
    validator.validate_records(records=records, target_usage=TargetUsageScope.RESEARCH)
    for rec, orig in zip(records, original_statuses):
        assert rec.rights_status == orig, f"Record '{rec.record_id}' was mutated from {orig} to {rec.rights_status}"


def test_legal_disclaimer_and_no_legal_certification_claims(validator: RightsValidationEngine) -> None:
    """Verify Rule 10: Reports state technical eligibility and explicitly disclaim legal certification or advice."""
    manifest = create_initial_registry_manifest()
    report = validator.validate_manifest(manifest=manifest, target_usage=TargetUsageScope.EVALUATION)

    assert report.legal_disclaimer is not None
    assert len(report.legal_disclaimer.strip()) > 0
    # Must disclaim legal certification / legal advice
    assert "does not constitute" in report.legal_disclaimer.lower()
    assert "legal advice" in report.legal_disclaimer.lower()
    assert "legal certification" in report.legal_disclaimer.lower()


def test_rejected_record_and_revoked_consent(validator: RightsValidationEngine) -> None:
    """Verify that explicitly REJECTED records and REVOKED consent are unconditionally blocked across all workloads."""
    # 1. Explicitly REJECTED record
    rec_rejected = DatasetImageRecord(
        record_id="SH-REJ-001",
        image_path_or_ref="safeher://tainted/bad_source.jpg",
        sha256=_make_dummy_hash("rejected_asset_001"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Tainted Collection",
        source_creator="Unknown",
        original_license="Apache-2.0",
        rights_status=RightsApprovalStatus.REJECTED,
        split=DatasetSplit.TEST,
    )

    report_rej_res = validator.validate_records([rec_rejected], target_usage=TargetUsageScope.RESEARCH)
    assert report_rej_res.is_valid is False
    assert "ERR_REJECTED_RECORD" in [v.rule_code for v in report_rej_res.violations]

    report_rej_eval = validator.validate_records([rec_rejected], target_usage=TargetUsageScope.EVALUATION)
    assert report_rej_eval.is_valid is False
    assert "ERR_REJECTED_RECORD" in [v.rule_code for v in report_rej_eval.violations]

    # 2. REVOKED consent record
    rec_revoked = DatasetImageRecord(
        record_id="SH-REVOKED-001",
        image_path_or_ref="safeher://internal/studio/revoked_subject.jpg",
        sha256=_make_dummy_hash("revoked_subject_asset"),
        label=GroundTruthLabel.AUTHENTIC,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="First-Party Consent-Based Photographs",
        source_creator="SafeHER Internal Studio",
        original_license="SafeHER-First-Party-Commercial",
        human_subjects_present=True,
        consent_reference="CR-SAFEHER-2026-F999",
        consent_status=ConsentStatus.REVOKED,  # Revoked by participant!
        split=DatasetSplit.TEST,
    )

    report_revoked = validator.validate_records([rec_revoked], target_usage=TargetUsageScope.RESEARCH)
    assert report_revoked.is_valid is False
    assert "ERR_REVOKED_CONSENT" in [v.rule_code for v in report_revoked.violations]


def test_candidate_registry_profile_complete_dimensions() -> None:
    """Verify that all 9 candidates have complete legally vetted metadata across all required dimensions."""
    candidates = list_all_candidates()
    assert len(candidates) == 9

    for profile in candidates:
        assert len(profile.dataset_id.strip()) > 0
        assert len(profile.name.strip()) > 0
        assert len(profile.source_url.strip()) > 0
        assert profile.source_url.startswith("http") or profile.source_url.startswith("safeher://")
        assert len(profile.reviewed_version.strip()) > 0
        assert len(profile.review_date.strip()) == 10  # YYYY-MM-DD
        assert len(profile.primary_license.strip()) > 0
        assert len(profile.terms_of_use_summary.strip()) > 10
        assert len(profile.upstream_restrictions.strip()) > 10
        assert len(profile.derivative_work_restrictions.strip()) > 10
        assert isinstance(profile.commercial_training_permitted, bool)
        assert len(profile.weight_redistribution_conditions.strip()) > 10
        assert len(profile.consent_withdrawal_policy.strip()) > 10
        assert profile.rights_status in RightsApprovalStatus
        assert len(profile.allowed_usage) > 0
        assert len(profile.unresolved_approval_requirements) > 0
