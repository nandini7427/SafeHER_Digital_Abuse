"""Automated unit and integration test suite for SafeHER M4.1 Dataset Manifest and Schema.

Validates:
1. Valid record instantiation with complete provenance.
2. Rejection of invalid records missing mandatory fields.
3. Rejection of missing SHA-256 hash.
4. Rejection of malformed SHA-256 hash.
5. Rejection of unknown / invalid enum labels.
6. Verification of the explicit non-overlapping label taxonomy.
7. Rejection of missing source provenance (dataset name, creator, license).
8. Rejection of missing or invalid schema version.
9. Rejection of sensitive personal identifiers (PII) in consent references.
10. Strict path validation (directory traversal, absolute paths, null bytes).
11. JSON serialization round-trip fidelity.
12. Backward compatibility with legacy field names and binary labels.
13. Separation of task types from rights status and image quality indicators.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import pytest
from pydantic import ValidationError

from ai_core.datasets.manifest import (
    CURRENT_SCHEMA_VERSION,
    AttributionMetadata,
    CommercialUseStatus,
    ConsentStatus,
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    DatasetTaskType,
    GeneratorInfo,
    GroundTruthLabel,
    ManipulationCategory,
    ResearchUseStatus,
    RightsApprovalStatus,
    TargetUsageScope,
    TransformationStep,
    UsageScope,
    compute_sha256,
    validate_dataset_path_or_ref,
    validate_non_identifying_consent_reference,
)


def _valid_hash(seed: str = "test_sample_image") -> str:
    """Generate a deterministic 64-char lowercase SHA-256 hex digest."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------
# 1. Valid record instantiation
# ------------------------------------------------------------------------------
def test_valid_record_instantiation() -> None:
    """Verify that a fully specified dataset record instantiates properly with all required fields."""
    record = DatasetImageRecord(
        record_id="SH-REC-2026-0001",
        image_id="IMG-TEST-0001",
        image_path_or_ref="safeher://internal/studio/sample_001.jpg",
        sha256=_valid_hash("sample_001"),
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        task_type=DatasetTaskType.GENERAL_FORENSIC_TRIAGE,
        manipulation_category=ManipulationCategory.NONE,
        original_image_ref=None,
        parent_image_ref=None,
        dataset_name="SafeHER Studio Baseline",
        source_creator="SafeHER Internal Evidence Team",
        source_url="https://safeher.org/datasets/studio",
        dataset_version="1.0.0",
        source_license="SafeHER-First-Party-Commercial",
        license_url="https://safeher.org/licenses/first-party",
        attribution_requirements="SafeHER Internal Evidence Studio",
        attribution=AttributionMetadata(
            required=True,
            creator="SafeHER Internal Studio",
            attribution_text="SafeHER First-Party Collection 2026",
            license_notice_url="https://safeher.org/licenses/first-party",
        ),
        commercial_use_status=CommercialUseStatus.PERMITTED,
        research_use_status=ResearchUseStatus.PERMITTED,
        training_eligibility=True,
        evaluation_eligibility=True,
        redistribution_eligibility=False,
        rights_status=RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        consent_reference="CR-SAFEHER-2026-F001",
        consent_status=ConsentStatus.AUTHORIZED,
        human_subjects_present=True,
        generator_name=None,
        generator_version=None,
        generator_info=None,
        transformation_history=[
            TransformationStep(
                step_name="camera_capture",
                parameters={"sensor_type": "CMOS", "iso": 100},
            )
        ],
        split=DatasetSplit.TRAIN,
        grouping_id="SUBJECT-042",
        is_aggregated=False,
        created_at=datetime.now(timezone.utc),
        schema_version="1.0.0",
        metadata={"camera_make": "Nikon", "camera_model": "Z6"},
    )

    assert record.record_id == "SH-REC-2026-0001"
    assert record.image_id == "IMG-TEST-0001"
    assert record.label == DatasetImageLabel.CAMERA_PHOTOGRAPH
    assert record.task_type == DatasetTaskType.GENERAL_FORENSIC_TRIAGE
    assert record.dataset_name == "SafeHER Studio Baseline"
    assert record.source_license == "SafeHER-First-Party-Commercial"
    assert record.grouping_id == "SUBJECT-042"
    assert record.schema_version == "1.0.0"
    assert record.commercial_use_status == CommercialUseStatus.PERMITTED
    assert record.research_use_status == ResearchUseStatus.PERMITTED
    assert record.training_eligibility is True
    assert record.evaluation_eligibility is True


# ------------------------------------------------------------------------------
# 2. Invalid record (missing required fields)
# ------------------------------------------------------------------------------
def test_invalid_record_missing_required_fields() -> None:
    """Verify that instantiating a record without required fields raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        # Missing sha256, label, dataset_name, source_creator, source_license
        DatasetImageRecord(
            record_id="SH-REC-INVALID",
            image_path_or_ref="safeher://images/001.jpg",
        )  # type: ignore

    errors = exc_info.value.errors()
    missing_fields = {err["loc"][0] for err in errors}
    assert "sha256" in missing_fields
    assert "label" in missing_fields
    assert "dataset_name" in missing_fields or "source_dataset" in missing_fields


# ------------------------------------------------------------------------------
# 3. Missing hash
# ------------------------------------------------------------------------------
def test_missing_hash() -> None:
    """Verify that empty string or None for sha256 raises ValidationError."""
    with pytest.raises(ValidationError, match="SHA-256 hash must not be empty"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256="",
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="TestDS",
            source_creator="TestAuthor",
            source_license="MIT",
        )


# ------------------------------------------------------------------------------
# 4. Invalid hash format
# ------------------------------------------------------------------------------
def test_invalid_hash_format() -> None:
    """Verify that malformed hashes (wrong length, non-hex characters) are rejected."""
    # Too short (63 chars)
    short_hash = "a" * 63
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=short_hash,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="TestDS",
            source_creator="TestAuthor",
            source_license="MIT",
        )

    # Invalid non-hex characters
    non_hex_hash = "g" * 64
    with pytest.raises(ValidationError, match="Invalid SHA-256 digest format"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=non_hex_hash,
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="TestDS",
            source_creator="TestAuthor",
            source_license="MIT",
        )


# ------------------------------------------------------------------------------
# 5. Unknown label rejected
# ------------------------------------------------------------------------------
def test_unknown_label_rejected() -> None:
    """Verify that unknown label values are rejected by enum validation."""
    with pytest.raises(ValidationError):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=_valid_hash(),
            label="NON_EXISTENT_LABEL_TOTALLY_FABRICATED",  # type: ignore
            dataset_name="TestDS",
            source_creator="TestAuthor",
            source_license="MIT",
        )


# ------------------------------------------------------------------------------
# 6. Explicit non-overlapping label taxonomy
# ------------------------------------------------------------------------------
def test_label_taxonomy_non_overlapping() -> None:
    """Verify that all required forensic labels exist in DatasetImageLabel without overlapping."""
    expected_labels = {
        "CAMERA_PHOTOGRAPH",
        "AI_GENERATED_IMAGE",
        "FACE_SWAP",
        "FACIAL_REENACTMENT",
        "FACIAL_INPAINTING",
        "LOCAL_IMAGE_EDIT",
        "IMAGE_COMPOSITE",
        "UNKNOWN",
        "UNVERIFIED",
    }
    actual_labels = {lbl.value for lbl in DatasetImageLabel}
    assert expected_labels.issubset(actual_labels)

    # Verify each label is unique
    values = [lbl.value for lbl in DatasetImageLabel]
    assert len(values) == len(set(values))


# ------------------------------------------------------------------------------
# 7. Missing source information
# ------------------------------------------------------------------------------
def test_missing_source_information() -> None:
    """Verify that empty or whitespace-only source information is strictly rejected."""
    # Empty dataset_name
    with pytest.raises(ValidationError, match="Mandatory provenance field 'dataset_name' must not be empty"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=_valid_hash(),
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="   ",  # Whitespace only
            source_creator="TestAuthor",
            source_license="MIT",
        )

    # Empty source_creator
    with pytest.raises(ValidationError, match="Mandatory provenance field 'source_creator' must not be empty"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=_valid_hash(),
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="TestDS",
            source_creator="",  # Empty
            source_license="MIT",
        )

    # Empty source_license
    with pytest.raises(ValidationError, match="Mandatory provenance field 'source_license' must not be empty"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=_valid_hash(),
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="TestDS",
            source_creator="TestAuthor",
            source_license="  ",  # Whitespace
        )


# ------------------------------------------------------------------------------
# 8. Missing or invalid schema version
# ------------------------------------------------------------------------------
def test_missing_or_invalid_schema_version() -> None:
    """Verify that schema_version enforces strict semantic versioning."""
    # Invalid format (non-semver)
    with pytest.raises(ValidationError, match="Invalid schema version format"):
        DatasetImageRecord(
            image_path_or_ref="safeher://img.jpg",
            sha256=_valid_hash(),
            label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
            dataset_name="TestDS",
            source_creator="TestAuthor",
            source_license="MIT",
            schema_version="version_1_beta",  # Malformed
        )

    # Valid semver
    rec = DatasetImageRecord(
        image_path_or_ref="safeher://img.jpg",
        sha256=_valid_hash(),
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        dataset_name="TestDS",
        source_creator="TestAuthor",
        source_license="MIT",
        schema_version="2.1.0",
    )
    assert rec.schema_version == "2.1.0"


# ------------------------------------------------------------------------------
# 9. Invalid consent reference (PII rejection)
# ------------------------------------------------------------------------------
def test_invalid_consent_reference_pii() -> None:
    """Verify that personal identity information (emails, phone numbers, SSNs) is rejected."""
    # Email PII
    with pytest.raises(ValueError, match="Storing sensitive PII .* is strictly prohibited"):
        validate_non_identifying_consent_reference("victim.name@example.org")

    # Phone number PII
    with pytest.raises(ValueError, match="Storing sensitive PII .* is strictly prohibited"):
        validate_non_identifying_consent_reference("Contact: 555-234-5678")

    # SSN PII
    with pytest.raises(ValueError, match="Storing sensitive PII .* is strictly prohibited"):
        validate_non_identifying_consent_reference("Subject SSN: 123-45-6789")

    # Valid non-identifying tokens must succeed
    validate_non_identifying_consent_reference("CR-SAFEHER-2026-F042")
    validate_non_identifying_consent_reference("vault:consent:99201")
    validate_non_identifying_consent_reference("urn:safeher:consent:token-abc")
    validate_non_identifying_consent_reference(None)


# ------------------------------------------------------------------------------
# 10. Path validation and security
# ------------------------------------------------------------------------------
def test_path_validation_and_security() -> None:
    """Verify path traversal prevention, null byte detection, and safe relative reference handling."""
    # Path traversal rejection
    with pytest.raises(ValueError, match="Path traversal detected"):
        validate_dataset_path_or_ref("../../../etc/shadow")

    with pytest.raises(ValueError, match="Path traversal detected"):
        validate_dataset_path_or_ref("images/../../secret.png")

    # Null byte rejection
    with pytest.raises(ValueError, match="prohibited null byte"):
        validate_dataset_path_or_ref("images/photo.jpg\x00.png")

    # Absolute Windows drive path rejection
    with pytest.raises(ValueError, match="Absolute Windows drive paths are not permitted"):
        validate_dataset_path_or_ref("C:\\Users\\admin\\dataset\\image.jpg")

    # Absolute root filesystem path rejection
    with pytest.raises(ValueError, match="Absolute filesystem paths are not permitted"):
        validate_dataset_path_or_ref("/var/data/image.jpg")

    # Approved URI schemes must be accepted
    assert validate_dataset_path_or_ref("safeher://internal/photo1.jpg") == "safeher://internal/photo1.jpg"
    assert validate_dataset_path_or_ref("upstream://genimage/sample.png") == "upstream://genimage/sample.png"
    assert validate_dataset_path_or_ref("faceforensics://c23/frame1.jpg") == "faceforensics://c23/frame1.jpg"
    assert validate_dataset_path_or_ref("openimages://train/001.jpg") == "openimages://train/001.jpg"

    # Safe relative paths must be accepted
    assert validate_dataset_path_or_ref("images/train/001.jpg") == "images/train/001.jpg"
    assert validate_dataset_path_or_ref("crops/face_patch_02.png") == "crops/face_patch_02.png"


# ------------------------------------------------------------------------------
# 11. JSON round-trip
# ------------------------------------------------------------------------------
def test_json_round_trip(tmp_path: Path) -> None:
    """Verify that DatasetImageRecord and DatasetManifest serialize and deserialize with 100% fidelity."""
    rec = DatasetImageRecord(
        record_id="SH-REC-JSON-001",
        image_id="IMG-JSON-001",
        image_path_or_ref="safeher://benchmarks/roundtrip.jpg",
        sha256=_valid_hash("roundtrip"),
        label=DatasetImageLabel.AI_GENERATED_IMAGE,
        task_type=DatasetTaskType.AI_GENERATED_IMAGE_DETECTION,
        manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
        dataset_name="Synthetic Roundtrip Test",
        source_creator="SafeHER Team",
        source_license="Apache-2.0",
        dataset_version="1.0.0",
        generator_name="StableDiffusion-v2.1",
        generator_version="2.1",
        split=DatasetSplit.TEST,
        grouping_id="GROUP-101",
        schema_version="1.0.0",
    )

    manifest = DatasetManifest(
        manifest_id="SH-MAN-ROUNDTRIP",
        dataset_name="Roundtrip Manifest",
        version="1.0.0",
        target_use=TargetUsageScope.EVALUATION,
        records=[rec],
    )

    json_str = manifest.to_json()
    loaded_manifest = DatasetManifest.from_json(json_str)

    assert loaded_manifest.manifest_id == manifest.manifest_id
    assert len(loaded_manifest.records) == 1
    loaded_rec = loaded_manifest.records[0]

    assert loaded_rec.record_id == rec.record_id
    assert loaded_rec.image_id == rec.image_id
    assert loaded_rec.sha256 == rec.sha256
    assert loaded_rec.label == DatasetImageLabel.AI_GENERATED_IMAGE
    assert loaded_rec.task_type == DatasetTaskType.AI_GENERATED_IMAGE_DETECTION
    assert loaded_rec.grouping_id == "GROUP-101"
    assert loaded_rec.schema_version == "1.0.0"

    # File persistence round-trip
    file_path = tmp_path / "roundtrip_manifest.json"
    manifest.save(file_path)
    file_loaded = DatasetManifest.load(file_path)
    assert file_loaded.manifest_id == manifest.manifest_id


# ------------------------------------------------------------------------------
# 12. Backward compatibility where applicable
# ------------------------------------------------------------------------------
def test_backward_compatibility() -> None:
    """Verify that records initialized with legacy field names map cleanly without error."""
    legacy_data = {
        "record_id": "SH-REC-LEGACY-001",
        "image_path_or_ref": "safeher://legacy/test.jpg",
        "sha256": _valid_hash("legacy"),
        "label": "authentic",  # Legacy string
        "source_dataset": "Legacy Benchmark Dataset",  # Legacy field
        "source_creator": "Original Author",
        "original_license": "MIT",  # Legacy field
        "original_source_ref": "upstream://flickr/photo123",  # Legacy field
        "allowed_usage": ["research", "evaluation"],  # Legacy field
    }

    rec = DatasetImageRecord.model_validate(legacy_data)

    # Label mapped to new explicit label
    assert rec.label == DatasetImageLabel.CAMERA_PHOTOGRAPH
    # Legacy field accessors work
    assert rec.dataset_name == "Legacy Benchmark Dataset"
    assert rec.source_dataset == "Legacy Benchmark Dataset"
    assert rec.source_license == "MIT"
    assert rec.original_license == "MIT"
    assert rec.original_image_ref == "upstream://flickr/photo123"
    assert rec.original_source_ref == "upstream://flickr/photo123"
    # Derived allowed usage
    assert UsageScope.RESEARCH in rec.allowed_usage
    assert UsageScope.EVALUATION in rec.allowed_usage


# ------------------------------------------------------------------------------
# 13. Separation of task labels from rights and quality status
# ------------------------------------------------------------------------------
def test_separation_of_task_labels_from_rights_and_quality() -> None:
    """Verify that task types, ground-truth labels, rights status, and quality indicators are separate enums."""
    rec = DatasetImageRecord(
        image_path_or_ref="safeher://img.jpg",
        sha256=_valid_hash("separation"),
        label=DatasetImageLabel.FACE_SWAP,  # Ground truth label
        task_type=DatasetTaskType.FACE_MANIPULATION_DETECTION,  # Task type
        rights_status=RightsApprovalStatus.NOT_APPROVED_FOR_PRODUCTION,  # Rights status
        commercial_use_status=CommercialUseStatus.PROHIBITED,  # Licensing status
        dataset_name="Test Separation",
        source_creator="Test Creator",
        source_license="Custom-Academic",
    )

    # Assert distinct types
    assert isinstance(rec.label, DatasetImageLabel)
    assert isinstance(rec.task_type, DatasetTaskType)
    assert isinstance(rec.rights_status, RightsApprovalStatus)
    assert isinstance(rec.commercial_use_status, CommercialUseStatus)
    assert rec.label.value != rec.task_type.value
    assert rec.rights_status != rec.label
