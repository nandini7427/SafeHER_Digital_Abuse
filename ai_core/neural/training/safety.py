"""Safety, rights validation, and split leakage enforcement for neural training pipelines.

Guarantees that:
1. Only approved training manifests are admitted into training pipelines.
2. Research-only or prohibited records are rejected in commercial/production training mode.
3. Test split data (internal test, held-out generators, cross-dataset) is strictly quarantined.
4. Forbidden lineages and identities do not leak across splits.
5. Private survivor case evidence is unconditionally quarantined from training.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.leakage import LeakageAuditReport, LeakageDetector, LeakageViolation
from ai_core.datasets.manifest import (
    CommercialUseStatus,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    RightsApprovalStatus,
    TargetUsageScope,
)
from ai_core.datasets.rights_validator import RightsValidationEngine, ValidationReport


@dataclass
class ManifestSafetyAudit:
    """Audit report produced prior to neural training pipeline execution."""

    is_safe: bool
    training_manifest_hash: str
    split_manifest_hash: str
    dataset_versions: List[str]
    train_record_count: int
    val_record_count: int
    test_record_count: int
    commercial_approved: bool
    quarantined_count: int
    notes: List[str]


def compute_records_hash(records: List[DatasetImageRecord]) -> str:
    """Compute deterministic cryptographic SHA-256 digest of record IDs and image hashes."""
    hasher = hashlib.sha256()
    sorted_recs = sorted(records, key=lambda r: r.record_id)
    for r in sorted_recs:
        record_bytes = f"{r.record_id}:{r.image_id}:{r.sha256}:{r.label.value}:{r.split.value}".encode("utf-8")
        hasher.update(record_bytes)
    return hasher.hexdigest()


def compute_split_allocation_hash(records: List[DatasetImageRecord]) -> str:
    """Compute deterministic SHA-256 digest of split allocations."""
    hasher = hashlib.sha256()
    sorted_recs = sorted(records, key=lambda r: r.record_id)
    for r in sorted_recs:
        split_bytes = f"{r.record_id}:{r.split.value}:{r.grouping_id or ''}".encode("utf-8")
        hasher.update(split_bytes)
    return hasher.hexdigest()


def enforce_training_safety_and_rights(
    manifest: DatasetManifest,
    target_workload: TargetUsageScope = TargetUsageScope.PRODUCTION_TRAINING,
    rights_engine: Optional[RightsValidationEngine] = None,
) -> ManifestSafetyAudit:
    """Validate dataset manifest against SafeHER training safety, rights, and leakage requirements.

    Args:
        manifest: Target DatasetManifest to validate.
        target_workload: TargetUsageScope (PRODUCTION_TRAINING or RESEARCH_EXPERIMENTATION).
        rights_engine: Optional RightsValidationEngine instance.

    Returns:
        ManifestSafetyAudit descriptor if all safety gates pass.

    Raises:
        ValueError: If rights violations, test-set leakage, or quarantined private data is detected.
    """
    engine = rights_engine or RightsValidationEngine()
    records = manifest.records

    notes: List[str] = []

    # 1. Private Survivor Evidence Quarantine Check
    for r in records:
        path_lower = (r.image_path_or_ref or "").lower()
        if "cases/" in path_lower or "survivor" in path_lower or "evidence/" in path_lower:
            raise ValueError(
                f"CRITICAL SAFETY VIOLATION: Private survivor case evidence detected in manifest! "
                f"Record '{r.record_id}' ({r.image_path_or_ref}) is strictly quarantined from all training pipelines."
            )

    # 2. Rights and Commercial-Training Validation
    if target_workload == TargetUsageScope.PRODUCTION_TRAINING:
        train_val_recs = [r for r in records if r.split in (DatasetSplit.TRAIN, DatasetSplit.VAL)]
        validation_report = engine.validate_records(train_val_recs, target_usage=target_workload)
        if not validation_report.is_valid:
            violations = [
                f"{v.record_id}: {v.message}"
                for v in validation_report.violations
            ]
            viol_sample = "; ".join(violations[:3])
            raise ValueError(
                f"Commercial Training Safety Gate Failed! Found {validation_report.failed_records} "
                f"unapproved or research-only records: {viol_sample}"
            )
        notes.append("Rights verification passed for production commercial training.")
    else:
        notes.append(f"Research experimentation mode permitted for workload {target_workload.value}.")

    # 3. Test Split Leakage & Partition Validation
    test_splits = {
        DatasetSplit.TEST,
        DatasetSplit.INTERNAL_TEST,
        DatasetSplit.HELD_OUT_GENERATOR_TEST,
        DatasetSplit.CROSS_DATASET_TEST,
        DatasetSplit.COMPRESSION_ROBUSTNESS_TEST,
    }

    train_recs = [r for r in records if r.split == DatasetSplit.TRAIN]
    val_recs = [r for r in records if r.split == DatasetSplit.VAL]
    test_recs = [r for r in records if r.split in test_splits]

    # 3a. Cryptographic & Path Duplicate Check
    test_sha256s: Set[str] = {r.sha256.lower() for r in test_recs if r.sha256}
    test_paths: Set[str] = {r.image_path_or_ref.strip().lower() for r in test_recs if r.image_path_or_ref}

    for r in train_recs + val_recs:
        if r.sha256 and r.sha256.lower() in test_sha256s:
            raise ValueError(
                f"Test Leakage Violation! Exact cryptographic duplicate image (SHA-256={r.sha256[:16]}...) "
                f"for record '{r.record_id}' ({r.split.value}) is present in the test set."
            )
        if r.image_path_or_ref and r.image_path_or_ref.strip().lower() in test_paths:
            raise ValueError(
                f"Test Leakage Violation! Image asset path '{r.image_path_or_ref}' "
                f"for record '{r.record_id}' ({r.split.value}) is present in the test set."
            )

    # 3b. Identity & Consent Reference Check
    test_groups: Set[str] = {r.grouping_id for r in test_recs if r.grouping_id}
    test_consents: Set[str] = {r.consent_reference for r in test_recs if r.consent_reference}

    for r in train_recs + val_recs:
        if r.grouping_id and r.grouping_id in test_groups:
            raise ValueError(
                f"Test Leakage Violation! {r.split.value} record '{r.record_id}' shares grouping_id '{r.grouping_id}' with test set."
            )
        if r.consent_reference and r.consent_reference in test_consents:
            raise ValueError(
                f"Test Leakage Violation! {r.split.value} record '{r.record_id}' shares consent_reference '{r.consent_reference}' with test set."
            )

    # 3c. Bidirectional Lineage / Parent-Child Check
    train_val_direct_ids: Set[str] = set()
    for r in train_recs + val_recs:
        train_val_direct_ids.add(r.record_id)
        if r.image_id:
            train_val_direct_ids.add(r.image_id)
        if r.image_path_or_ref:
            train_val_direct_ids.add(r.image_path_or_ref.strip().lower())

    test_direct_ids: Set[str] = set()
    for r in test_recs:
        test_direct_ids.add(r.record_id)
        if r.image_id:
            test_direct_ids.add(r.image_id)
        if r.image_path_or_ref:
            test_direct_ids.add(r.image_path_or_ref.strip().lower())

    # Check if any test record derives from a train/val parent
    for r in test_recs:
        for pref in (r.original_image_ref, r.parent_image_ref):
            if pref and (pref in train_val_direct_ids or pref.strip().lower() in train_val_direct_ids):
                raise ValueError(
                    f"Test Leakage Violation! Test record '{r.record_id}' is a derived variant of "
                    f"training/validation parent image '{pref}'."
                )

    # Check if any train/val record derives from a test parent
    for r in train_recs + val_recs:
        for pref in (r.original_image_ref, r.parent_image_ref):
            if pref and (pref in test_direct_ids or pref.strip().lower() in test_direct_ids):
                raise ValueError(
                    f"Test Leakage Violation! {r.split.value} record '{r.record_id}' is a derived variant of "
                    f"test parent image '{pref}'."
                )

    # Check if train and test share any common root/parent references
    test_original_images: Set[str] = {r.original_image_ref.strip().lower() for r in test_recs if r.original_image_ref}
    test_parent_images: Set[str] = {r.parent_image_ref.strip().lower() for r in test_recs if r.parent_image_ref}

    for r in train_recs:
        if r.original_image_ref and r.original_image_ref.strip().lower() in test_original_images:
            raise ValueError(
                f"Test Leakage Violation! Train record '{r.record_id}' shares original_image_ref '{r.original_image_ref}' with test set."
            )
        if r.parent_image_ref and r.parent_image_ref.strip().lower() in test_parent_images:
            raise ValueError(
                f"Test Leakage Violation! Train record '{r.record_id}' shares parent_image_ref '{r.parent_image_ref}' with test set."
            )

    # 4. Compute cryptographic hashes
    manifest_hash = compute_records_hash(train_recs + val_recs)
    split_hash = compute_split_allocation_hash(records)
    versions = sorted(list({r.dataset_version for r in records if r.dataset_version}))

    return ManifestSafetyAudit(
        is_safe=True,
        training_manifest_hash=manifest_hash,
        split_manifest_hash=split_hash,
        dataset_versions=versions,
        train_record_count=len(train_recs),
        val_record_count=len(val_recs),
        test_record_count=len(test_recs),
        commercial_approved=(target_workload == TargetUsageScope.PRODUCTION_TRAINING),
        quarantined_count=0,
        notes=notes,
    )
