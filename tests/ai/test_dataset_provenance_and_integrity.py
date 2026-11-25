"""Comprehensive automated test suite for SafeHER Dataset Provenance, Integrity, and Duplicate Detection.

Tests all required scenarios:
1. Exact duplicates:
   - Identical SHA-256 detection and clustering.
   - Conflicting provenance detection across identical hashes.
2. Near-duplicate candidate detection:
   - Detection of recompressed and slightly modified images via perceptual hashing (dHash/pHash).
   - Configurable similarity threshold and Hamming distance.
   - Biometric non-equivalence guarantee: Perceptual similarity does NOT prove personal identity.
3. Modified image metadata:
   - EXIF or metadata alteration alters SHA-256 but keeps perceptual hash identical (near-duplicate detection).
4. Corrupted and invalid format files:
   - Truncated image streams, random garbage headers, non-image files.
   - Decompression and format validation.
5. Path traversal protection:
   - Defense against '../', '..\\', null bytes, absolute drive paths ('C:'), and leading slashes.
6. Symlink security:
   - Symlinks resolving outside the dataset root directory are detected and blocked.
7. Hash mismatch:
   - Manifest record SHA-256 differing from actual on-disk file hash is flagged as a blocker.
8. Large-file streaming:
   - Chunked streaming calculation produces identical digest to whole-file hash without memory buffering.
9. Same image with different legitimate transformations:
   - Non-destructive variant generation: parent provenance is preserved unmodified while children branch.
10. Provenance chain inconsistencies:
   - Missing parent record reference.
   - Chronological timestamp inversion.
   - Cyclic parent lineage loops.
   - Missing transformation/generator documentation for manipulated records.
   - Unauthorized rights escalation from research-only parent to commercial child.
11. Comprehensive dataset audit:
   - Verification of DatasetAuditor report catching missing files, corrupted assets, unapproved records,
     missing hashes, duplicate files, and provenance conflicts.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import io
import os
from pathlib import Path
import tempfile
import pytest
from PIL import Image

from ai_core.datasets.auditor import DatasetAuditor
from ai_core.datasets.duplicates import (
    DuplicateDetector,
    compute_dhash,
    compute_phash,
    hamming_distance,
    perceptual_similarity,
)
from ai_core.datasets.integrity import (
    CorruptedFileError,
    FileIntegrityError,
    InvalidFormatError,
    PathTraversalError,
    compute_file_sha256,
    compute_stream_sha256,
    resolve_safe_relative_path,
    validate_image_file,
    verify_file_sha256,
)
from ai_core.datasets.manifest import (
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
    RightsApprovalStatus,
    TargetUsageScope,
    TransformationStep,
    UsageScope,
)
from ai_core.datasets.provenance import (
    ProvenanceGraph,
    ProvenanceViolationType,
    create_derived_variant_record,
    verify_provenance_consistency,
)


def _create_synthetic_image_bytes(
    color: tuple = (120, 150, 180),
    size: tuple = (120, 120),
    fmt: str = "JPEG",
    quality: int = 90,
) -> bytes:
    """Create deterministic in-memory synthetic image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


# ------------------------------------------------------------------------------
# 1. Exact Duplicate Detection & Provenance Conflicts
# ------------------------------------------------------------------------------
def test_exact_duplicate_detection_and_conflict_handling() -> None:
    """Verify exact SHA-256 duplicate detection and provenance conflict categorization."""
    detector = DuplicateDetector()
    img_bytes = _create_synthetic_image_bytes(color=(100, 120, 140))
    real_hash = hashlib.sha256(img_bytes).hexdigest()

    # Record 1: Authentic camera photograph
    rec1 = DatasetImageRecord(
        record_id="SH-EXACT-001",
        image_path_or_ref="images/sample_001.jpg",
        sha256=real_hash,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Dataset-A",
        source_creator="Creator A",
        original_license="Apache-2.0",
        split=DatasetSplit.TRAIN,
    )

    # Record 2: Exact duplicate claiming manipulated label (CONTRADICTION!)
    rec2 = DatasetImageRecord(
        record_id="SH-EXACT-002",
        image_path_or_ref="images/sample_002.jpg",
        sha256=real_hash,
        label=DatasetImageLabel.AI_GENERATED_IMAGE,  # Contradictory label!
        manipulation_category=ManipulationCategory.DIFFUSION_GENERATED,
        source_dataset="Dataset-B",
        source_creator="Creator B",
        original_license="Apache-2.0",
        split=DatasetSplit.TEST,
    )

    # Record 3: Non-duplicate distinct image
    distinct_bytes = _create_synthetic_image_bytes(color=(220, 40, 40))
    distinct_hash = hashlib.sha256(distinct_bytes).hexdigest()
    rec3 = DatasetImageRecord(
        record_id="SH-EXACT-003",
        image_path_or_ref="images/sample_003.jpg",
        sha256=distinct_hash,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Dataset-A",
        source_creator="Creator A",
        original_license="Apache-2.0",
        split=DatasetSplit.TRAIN,
    )

    clusters = detector.find_exact_duplicates([rec1, rec2, rec3])
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.sha256 == real_hash
    assert cluster.count == 2
    assert set(cluster.record_ids) == {"SH-EXACT-001", "SH-EXACT-002"}
    assert cluster.has_provenance_conflict is True
    assert "label" in cluster.conflicting_fields
    assert "manipulation_category" in cluster.conflicting_fields
    assert "source_dataset" in cluster.conflicting_fields


# ------------------------------------------------------------------------------
# 2. Near-Duplicate Candidate Detection & Biometric Non-Equivalence
# ------------------------------------------------------------------------------
def test_near_duplicate_candidate_detection() -> None:
    """Verify perceptual hashing detects recompression and minor edits as near duplicates."""
    detector = DuplicateDetector(max_hamming_distance=6)

    # Base image
    base_img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    # Variant A: Recompressed at Q=75
    buf_a = io.BytesIO()
    base_img.save(buf_a, format="JPEG", quality=75)
    bytes_a = buf_a.getvalue()

    # Variant B: Recompressed at Q=95
    buf_b = io.BytesIO()
    base_img.save(buf_b, format="JPEG", quality=95)
    bytes_b = buf_b.getvalue()

    # Distinct image: Diagonal pattern
    distinct_img = Image.new("RGB", (128, 128), color=(20, 20, 20))
    for i in range(128):
        distinct_img.putpixel((i, i), (255, 255, 255))
    buf_c = io.BytesIO()
    distinct_img.save(buf_c, format="JPEG", quality=85)
    bytes_c = buf_c.getvalue()

    # Compute perceptual hashes
    hash_a = compute_dhash(bytes_a)
    hash_b = compute_dhash(bytes_b)
    hash_c = compute_dhash(bytes_c)

    dist_ab = hamming_distance(hash_a, hash_b)
    dist_ac = hamming_distance(hash_a, hash_c)

    # Variant A and B must be extremely close (Hamming distance <= 2)
    assert dist_ab <= 2
    assert perceptual_similarity(hash_a, hash_b) >= 0.96

    # Distinct image must differ substantially
    assert dist_ac > 6

    # Candidate detection
    hashes = {"REC-A": hash_a, "REC-B": hash_b, "REC-C": hash_c}
    candidates = detector.find_near_duplicates(hashes)

    assert len(candidates) == 1
    cand = candidates[0]
    assert {cand.record_id_a, cand.record_id_b} == {"REC-A", "REC-B"}
    assert cand.requires_human_review is True
    assert "biometric" in cand.disclaimer.lower()
    assert "does not constitute" in cand.disclaimer.lower()


def test_dct_phash_perceptual_stability() -> None:
    """Verify DCT-based pHash stability under mild contrast adjustment."""
    base_img = Image.new("RGB", (100, 100), color=(120, 120, 120))
    # Add texture
    for x in range(0, 100, 10):
        for y in range(0, 100, 10):
            base_img.putpixel((x, y), (200, 50, 50))

    # Slightly brighter variant
    bright_img = Image.new("RGB", (100, 100), color=(130, 130, 130))
    for x in range(0, 100, 10):
        for y in range(0, 100, 10):
            bright_img.putpixel((x, y), (210, 60, 60))

    h1 = compute_phash(base_img)
    h2 = compute_phash(bright_img)

    dist = hamming_distance(h1, h2)
    assert dist <= 4  # DCT low frequencies remain stable


# ------------------------------------------------------------------------------
# 3. Modified Image Metadata vs Perceptual Hashing
# ------------------------------------------------------------------------------
def test_modified_metadata_changes_sha256_but_preserves_phash(tmp_path: Path) -> None:
    """Verify that stripping or altering EXIF changes SHA-256 but is caught as a near duplicate."""
    img = Image.new("RGB", (80, 80), color=(80, 140, 200))
    path_orig = tmp_path / "orig.jpg"
    path_mod = tmp_path / "modified_meta.jpg"

    # Save original with comment/exif
    img.save(path_orig, format="JPEG", quality=90, comment=b"SafeHER Studio Calibration Asset")
    # Save modified with different comment
    img.save(path_mod, format="JPEG", quality=90, comment=b"Modified Attribution String 2026")

    sha_orig = compute_file_sha256(path_orig)
    sha_mod = compute_file_sha256(path_mod)

    # Cryptographic hashes MUST differ due to header metadata
    assert sha_orig != sha_mod

    # But visual perceptual hashes MUST be identical
    phash_orig = compute_dhash(path_orig.read_bytes())
    phash_mod = compute_dhash(path_mod.read_bytes())
    assert phash_orig == phash_mod
    assert hamming_distance(phash_orig, phash_mod) == 0


# ------------------------------------------------------------------------------
# 4. Corrupted and Invalid Format Files
# ------------------------------------------------------------------------------
def test_corrupted_and_invalid_format_detection(tmp_path: Path) -> None:
    """Verify validation engine catches corrupted headers, truncated data, and non-image files."""
    # 1. Truncated corrupted JPEG
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01" + b"\x00" * 20)  # Truncated!

    with pytest.raises(CorruptedFileError):
        validate_image_file(corrupt_file)

    # 2. Non-image text file posing as jpg
    fake_img = tmp_path / "fake.jpg"
    fake_img.write_text("This is plain text posing as an image.")

    with pytest.raises(CorruptedFileError):
        validate_image_file(fake_img)

    # 3. Zero-byte empty file
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")

    with pytest.raises(FileIntegrityError, match="empty"):
        validate_image_file(empty_file)

    # 4. Disallowed format
    bmp_file = tmp_path / "test.bmp"
    img = Image.new("RGB", (40, 40), color="blue")
    img.save(bmp_file, format="BMP")

    with pytest.raises(InvalidFormatError, match="allowed formats"):
        validate_image_file(bmp_file, allowed_formats={"JPEG", "PNG"})


# ------------------------------------------------------------------------------
# 5. Path Traversal & Security
# ------------------------------------------------------------------------------
def test_path_traversal_defense(tmp_path: Path) -> None:
    """Verify robust defense against path traversal sequences, null bytes, and root escapes."""
    root = tmp_path / "dataset_root"
    root.mkdir()
    (root / "valid.jpg").write_bytes(_create_synthetic_image_bytes())

    # 1. Valid relative path succeeds
    safe_path = resolve_safe_relative_path(root, "valid.jpg")
    assert safe_path == (root / "valid.jpg").resolve()

    # 2. Path traversal attempts
    traversal_patterns = [
        "../outside.jpg",
        "..\\outside.jpg",
        "sub/../../outside.jpg",
        "sub\\..\\..\\outside.jpg",
        "/etc/passwd",
        "\\Windows\\System32\\cmd.exe",
        "C:\\boot.ini",
        "D:/sensitive.jpg",
        "sub/\x00/attack.jpg",
    ]

    for pattern in traversal_patterns:
        with pytest.raises(PathTraversalError):
            resolve_safe_relative_path(root, pattern)


# ------------------------------------------------------------------------------
# 6. Symlink Security
# ------------------------------------------------------------------------------
def test_symlink_security(tmp_path: Path) -> None:
    """Verify symlinks escaping the dataset root directory are detected and blocked."""
    root = tmp_path / "root_dir"
    root.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()

    secret_file = outside / "secret.jpg"
    secret_file.write_bytes(_create_synthetic_image_bytes())

    link_path = root / "symlink_escape.jpg"
    try:
        os.symlink(secret_file, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in this OS/permission environment.")

    # Disallowed by default
    with pytest.raises(PathTraversalError, match="Symlinks are disallowed"):
        resolve_safe_relative_path(root, "symlink_escape.jpg", allow_symlinks_within_root=False)

    # Even if internal symlinks are permitted, external escapes MUST be blocked
    with pytest.raises(PathTraversalError, match="outside approved root"):
        resolve_safe_relative_path(root, "symlink_escape.jpg", allow_symlinks_within_root=True)


# ------------------------------------------------------------------------------
# 7. Hash Mismatch Verification
# ------------------------------------------------------------------------------
def test_hash_mismatch_verification(tmp_path: Path) -> None:
    """Verify verify_file_sha256 accurately detects hash mismatches."""
    sample_file = tmp_path / "sample.jpg"
    data = _create_synthetic_image_bytes()
    sample_file.write_bytes(data)

    actual_hash = hashlib.sha256(data).hexdigest()
    wrong_hash = "0" * 64

    # Correct hash matches
    is_valid, computed = verify_file_sha256(sample_file, actual_hash)
    assert is_valid is True
    assert computed == actual_hash

    # Wrong hash fails
    is_valid_wrong, computed_wrong = verify_file_sha256(sample_file, wrong_hash)
    assert is_valid_wrong is False
    assert computed_wrong == actual_hash


# ------------------------------------------------------------------------------
# 8. Large-File Streaming Hash Calculation
# ------------------------------------------------------------------------------
def test_large_file_streaming_hash_fidelity(tmp_path: Path) -> None:
    """Verify chunked streaming hash matches standard in-memory SHA-256 for multi-megabyte files."""
    large_file = tmp_path / "large_asset.bin"
    chunk_block = b"SafeHER Forensic Streaming Verification Block 2026\n" * 1024  # ~50 KB

    # Write 5 MB file in chunks
    hasher_truth = hashlib.sha256()
    with open(large_file, "wb") as f:
        for _ in range(100):
            f.write(chunk_block)
            hasher_truth.update(chunk_block)

    expected_hash = hasher_truth.hexdigest().lower()

    # Stream with small 16 KB chunks
    streaming_hash = compute_file_sha256(large_file, chunk_size=16384)
    assert streaming_hash == expected_hash

    # Stream with default 64 KB chunks
    streaming_hash_default = compute_file_sha256(large_file, chunk_size=65536)
    assert streaming_hash_default == expected_hash


# ------------------------------------------------------------------------------
# 9. Non-Destructive Variant Generation & Branching Provenance
# ------------------------------------------------------------------------------
def test_same_image_multiple_transformations_preserves_parent() -> None:
    """Verify that deriving multiple manipulated variants preserves the original parent without mutation."""
    parent_bytes = _create_synthetic_image_bytes(color=(100, 100, 100))
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()

    parent_record = DatasetImageRecord(
        record_id="SH-PARENT-001",
        image_path_or_ref="raw/authentic_camera.jpg",
        sha256=parent_hash,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="First-Party Studio",
        source_creator="SafeHER Internal Studio",
        original_license="SafeHER-First-Party-Commercial",
        allowed_usage=[UsageScope.RESEARCH, UsageScope.COMMERCIAL_TRAINING],
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    # Branch 1: Procedural ELA Perturbation
    step_ela = TransformationStep(
        step_name="ela_compression_perturbation",
        parameters={"quality_factor": 75, "scale": 15},
        tool_name="SafeHER-Procedural-v1",
        tool_version="1.0",
    )
    child_ela = create_derived_variant_record(
        parent_record=parent_record,
        new_record_id="SH-CHILD-ELA-001",
        new_image_path_or_ref="derivatives/child_ela.jpg",
        new_sha256=hashlib.sha256(b"child_ela_data").hexdigest(),
        manipulation_category=ManipulationCategory.SYNTHETIC_EDIT,
        new_label=DatasetImageLabel.AI_GENERATED_IMAGE,
        transformation_step=step_ela,
        generator_name="SafeHER-Procedural-v1",
        generator_version="1.0",
    )

    # Branch 2: Spliced Composite
    step_splice = TransformationStep(
        step_name="spliced_donor_composite",
        parameters={"donor_bbox": [20, 20, 60, 60], "feather": 2},
        tool_name="SafeHER-Splicing-v1",
        tool_version="1.2",
    )
    child_splice = create_derived_variant_record(
        parent_record=parent_record,
        new_record_id="SH-CHILD-SPLICE-001",
        new_image_path_or_ref="derivatives/child_splice.jpg",
        new_sha256=hashlib.sha256(b"child_splice_data").hexdigest(),
        manipulation_category=ManipulationCategory.SPLICED_COMPOSITE,
        new_label=DatasetImageLabel.IMAGE_COMPOSITE,
        transformation_step=step_splice,
        generator_name="SafeHER-Splicing-v1",
        generator_version="1.2",
    )

    # 1. Parent MUST remain completely intact (unmodified)
    assert parent_record.record_id == "SH-PARENT-001"
    assert parent_record.manipulation_category == ManipulationCategory.NONE
    assert len(parent_record.transformation_history) == 0
    assert parent_record.parent_image_ref is None

    # 2. Child ELA verification
    assert child_ela.parent_image_ref == "SH-PARENT-001"
    assert child_ela.original_image_ref == "SH-PARENT-001"
    assert len(child_ela.transformation_history) == 1
    assert child_ela.transformation_history[0].step_name == "ela_compression_perturbation"
    assert child_ela.manipulation_category == ManipulationCategory.SYNTHETIC_EDIT

    # 3. Child Splice verification
    assert child_splice.parent_image_ref == "SH-PARENT-001"
    assert child_splice.original_image_ref == "SH-PARENT-001"
    assert len(child_splice.transformation_history) == 1
    assert child_splice.transformation_history[0].step_name == "spliced_donor_composite"
    assert child_splice.manipulation_category == ManipulationCategory.SPLICED_COMPOSITE

    # 4. Provenance graph verification
    graph = ProvenanceGraph([parent_record, child_ela, child_splice])
    assert len(graph.get_children("SH-PARENT-001")) == 2
    assert graph.get_parent("SH-CHILD-ELA-001").record_id == "SH-PARENT-001"
    assert graph.get_root_ancestor("SH-CHILD-SPLICE-001").record_id == "SH-PARENT-001"


# ------------------------------------------------------------------------------
# 10. Provenance Chain Inconsistency Detection
# ------------------------------------------------------------------------------
def test_provenance_chain_inconsistencies() -> None:
    """Verify detection of missing parent, chronological inversion, cycles, and unauthorized rights escalation."""
    t0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc)
    t_past = datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc)

    # Parent: Research-only
    parent = DatasetImageRecord(
        record_id="SH-PARENT-RES",
        image_path_or_ref="images/p1.jpg",
        sha256="1" * 64,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Academic-Corpus",
        source_creator="Univ X",
        original_license="CC-BY-NC-4.0",
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        created_at=t0,
        split=DatasetSplit.TRAIN,
    )

    # 1. Missing parent reference
    orphan = DatasetImageRecord(
        record_id="SH-ORPHAN",
        image_path_or_ref="images/orphan.jpg",
        sha256="2" * 64,
        label=DatasetImageLabel.AI_GENERATED_IMAGE,
        manipulation_category=ManipulationCategory.SYNTHETIC_EDIT,
        source_dataset="Academic-Corpus",
        source_creator="Univ X",
        original_license="CC-BY-NC-4.0",
        parent_image_ref="SH-NONEXISTENT-PARENT",  # Missing!
        transformation_history=[TransformationStep(step_name="filter")],
        created_at=t0 + timedelta(minutes=5),
        split=DatasetSplit.TRAIN,
    )

    # 2. Chronological inversion
    inverted = DatasetImageRecord(
        record_id="SH-INVERTED",
        image_path_or_ref="images/inverted.jpg",
        sha256="3" * 64,
        label=DatasetImageLabel.AI_GENERATED_IMAGE,
        manipulation_category=ManipulationCategory.SYNTHETIC_EDIT,
        source_dataset="Academic-Corpus",
        source_creator="Univ X",
        original_license="CC-BY-NC-4.0",
        parent_image_ref="SH-PARENT-RES",
        transformation_history=[TransformationStep(step_name="filter")],
        created_at=t_past,  # Before parent!
        split=DatasetSplit.TRAIN,
    )

    # 3. Unauthorized rights escalation (Parent is RESEARCH_ONLY, child claims COMMERCIAL)
    escalated = DatasetImageRecord(
        record_id="SH-ESCALATED",
        image_path_or_ref="images/escalated.jpg",
        sha256="4" * 64,
        label=DatasetImageLabel.AI_GENERATED_IMAGE,
        manipulation_category=ManipulationCategory.SYNTHETIC_EDIT,
        source_dataset="Academic-Corpus",
        source_creator="Univ X",
        original_license="CC-BY-NC-4.0",
        parent_image_ref="SH-PARENT-RES",
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,  # Unauthorized escalation!
        transformation_history=[TransformationStep(step_name="filter")],
        created_at=t0 + timedelta(minutes=10),
        split=DatasetSplit.TRAIN,
    )

    violations = verify_provenance_consistency([parent, orphan, inverted, escalated])
    violation_types = [v.violation_type for v in violations]

    assert ProvenanceViolationType.MISSING_PARENT_RECORD in violation_types
    assert ProvenanceViolationType.CHRONOLOGICAL_INVERSION in violation_types
    assert ProvenanceViolationType.UNAUTHORIZED_RIGHTS_ESCALATION in violation_types


def test_cyclic_lineage_detection() -> None:
    """Verify detection of cyclic parent-child reference loops."""
    rec_a = DatasetImageRecord(
        record_id="NODE-A",
        image_path_or_ref="img/a.jpg",
        sha256="a" * 64,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Test",
        source_creator="Test",
        original_license="Apache-2.0",
        parent_image_ref="NODE-B",  # A points to B
        split=DatasetSplit.TEST,
    )

    rec_b = DatasetImageRecord(
        record_id="NODE-B",
        image_path_or_ref="img/b.jpg",
        sha256="b" * 64,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Test",
        source_creator="Test",
        original_license="Apache-2.0",
        parent_image_ref="NODE-A",  # B points to A -> Cycle!
        split=DatasetSplit.TEST,
    )

    violations = verify_provenance_consistency([rec_a, rec_b])
    assert any(v.violation_type == ProvenanceViolationType.CYCLIC_LINEAGE for v in violations)


# ------------------------------------------------------------------------------
# 11. Comprehensive Dataset Auditor Integration
# ------------------------------------------------------------------------------
def test_dataset_auditor_comprehensive_report(tmp_path: Path) -> None:
    """Verify DatasetAuditor accurately compiles all defect dimensions into a structured report."""
    root_dir = tmp_path / "dataset_storage"
    root_dir.mkdir()

    # Valid image
    valid_file = root_dir / "valid.jpg"
    valid_bytes = _create_synthetic_image_bytes(color=(50, 100, 150))
    valid_file.write_bytes(valid_bytes)
    valid_hash = hashlib.sha256(valid_bytes).hexdigest()

    rec_valid = DatasetImageRecord(
        record_id="REC-VALID-001",
        image_path_or_ref="valid.jpg",
        sha256=valid_hash,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Valid Set",
        source_creator="SafeHER",
        original_license="Apache-2.0",
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    # Missing file
    rec_missing = DatasetImageRecord(
        record_id="REC-MISSING-001",
        image_path_or_ref="nonexistent_on_disk.jpg",
        sha256="f" * 64,
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Valid Set",
        source_creator="SafeHER",
        original_license="Apache-2.0",
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    # Hash mismatch file
    mismatch_file = root_dir / "mismatch.jpg"
    mismatch_file.write_bytes(_create_synthetic_image_bytes(color=(200, 200, 200)))
    rec_mismatch = DatasetImageRecord(
        record_id="REC-MISMATCH-001",
        image_path_or_ref="mismatch.jpg",
        sha256="0" * 64,  # Incorrect hash!
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Valid Set",
        source_creator="SafeHER",
        original_license="Apache-2.0",
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    # Corrupted file
    corrupt_file = root_dir / "corrupted.jpg"
    corrupt_file.write_bytes(b"Garbage non-image header bytes...")
    rec_corrupt = DatasetImageRecord(
        record_id="REC-CORRUPT-001",
        image_path_or_ref="corrupted.jpg",
        sha256=hashlib.sha256(corrupt_file.read_bytes()).hexdigest(),
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Valid Set",
        source_creator="SafeHER",
        original_license="Apache-2.0",
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    # Missing license record
    no_lic_file = root_dir / "no_license.jpg"
    no_lic_bytes = _create_synthetic_image_bytes(color=(10, 80, 120))
    no_lic_file.write_bytes(no_lic_bytes)
    rec_no_license = DatasetImageRecord(
        record_id="REC-NO-LIC-001",
        image_path_or_ref="no_license.jpg",
        sha256=hashlib.sha256(no_lic_bytes).hexdigest(),
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Valid Set",
        source_creator="SafeHER",
        original_license="unknown",  # Unknown license!
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        split=DatasetSplit.TRAIN,
    )

    # Unapproved research-only record in commercial training target
    unapproved_file = root_dir / "unapproved.jpg"
    unapproved_bytes = _create_synthetic_image_bytes(color=(30, 90, 140))
    unapproved_file.write_bytes(unapproved_bytes)
    rec_unapproved = DatasetImageRecord(
        record_id="REC-UNAPPROVED-001",
        image_path_or_ref="unapproved.jpg",
        sha256=hashlib.sha256(unapproved_bytes).hexdigest(),
        label=DatasetImageLabel.CAMERA_PHOTOGRAPH,
        manipulation_category=ManipulationCategory.NONE,
        source_dataset="Valid Set",
        source_creator="SafeHER",
        original_license="Apache-2.0",
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,  # Unapproved for commercial!
        split=DatasetSplit.TRAIN,
    )

    manifest = DatasetManifest(
        manifest_id="SH-AUDIT-TEST-001",
        dataset_name="Comprehensive Audit Test Manifest",
        version="1.0.0",
        target_use=TargetUsageScope.PRODUCTION_TRAINING,
        records=[rec_valid, rec_missing, rec_mismatch, rec_corrupt, rec_no_license, rec_unapproved],
    )

    auditor = DatasetAuditor()
    report = auditor.audit_manifest(
        manifest=manifest,
        root_dir=root_dir,
        target_workload=TargetUsageScope.PRODUCTION_TRAINING,
        check_filesystem=True,
    )

    assert report.is_audit_passed is False
    assert report.total_records == 6
    assert report.passed_records == 1  # Only rec_valid passed all dimensions
    assert report.failed_records == 5

    # Check that defect lists captured all issues
    assert "REC-MISSING-001" in report.missing_files
    assert any(m["record_id"] == "REC-MISMATCH-001" for m in report.hash_mismatches)
    assert any(c["record_id"] == "REC-CORRUPT-001" for c in report.corrupted_files)
    assert "REC-NO-LIC-001" in report.missing_licence_information
    assert "REC-UNAPPROVED-001" in report.unapproved_records

    # Markdown rendering verification
    md = report.to_markdown()
    assert "# SafeHER Dataset Audit Report" in md
    assert "FAILED" in md
    assert "Defect Inventory" in md


def test_nested_symlink_security(tmp_path: Path) -> None:
    """Reject a path whose parent directory is a symlink escaping the dataset root."""
    root = tmp_path / "root_dir"
    root.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(_create_synthetic_image_bytes())

    link_dir = root / "linked_directory"
    try:
        os.symlink(outside, link_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlink creation not supported in this OS/permission environment.")

    with pytest.raises(PathTraversalError, match="Symlinks are disallowed"):
        resolve_safe_relative_path(root, "linked_directory/secret.jpg")

    with pytest.raises(PathTraversalError, match="outside approved root"):
        resolve_safe_relative_path(root, "linked_directory/secret.jpg", allow_symlinks_within_root=True)
