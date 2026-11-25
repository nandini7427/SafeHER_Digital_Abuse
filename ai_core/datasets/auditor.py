"""Dataset audit engine for SafeHER image datasets.

Generates comprehensive audit reports analyzing:
1. Missing files (records whose on-disk assets cannot be located).
2. Missing hashes (records lacking valid 64-char SHA-256 digests).
3. Hash mismatches (manifest SHA-256 differs from actual on-disk file).
4. Duplicate files (exact SHA-256 duplicates + near-duplicate perceptual candidates).
5. Conflicting provenance (identical hashes with conflicting labels/licenses/sources).
6. Invalid formats (unrecognized formats or disallowed image encodings).
7. Corrupted files (truncated, unreadable, or malformed image bitstreams).
8. Missing license information (unspecified, unknown, or empty upstream terms).
9. Unapproved records (research-only, pending, or unreviewed assets in commercial manifests).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.duplicates import (
    DuplicateDetector,
    ExactDuplicateCluster,
    NearDuplicateCandidate,
    compute_dhash,
)
from ai_core.datasets.integrity import (
    CorruptedFileError,
    FileIntegrityError,
    InvalidFormatError,
    PathTraversalError,
    compute_file_sha256,
    resolve_safe_relative_path,
    validate_image_file,
)
from ai_core.datasets.manifest import (
    DatasetImageRecord,
    DatasetManifest,
    RightsApprovalStatus,
    TargetUsageScope,
)
from ai_core.datasets.provenance import (
    ProvenanceViolation,
    verify_provenance_consistency,
)


class AuditFinding(BaseModel):
    """Individual audit observation or defect."""

    category: str = Field(..., description="Audit dimension (e.g. 'MISSING_FILE', 'CORRUPTED_FILE')")
    record_id: str = Field(..., description="Affected record ID")
    severity: str = Field(..., description="'BLOCKER', 'WARNING', or 'INFO'")
    details: str = Field(..., description="Specific defect description")
    remediation: str = Field(..., description="Actionable fix")

    model_config = ConfigDict(frozen=True)


class DatasetAuditReport(BaseModel):
    """Complete dataset audit report covering filesystem, cryptographic, and licensing integrity."""

    dataset_name: str = Field(..., description="Audited dataset or manifest title")
    manifest_id: Optional[str] = Field(None, description="Manifest identifier if available")
    target_workload: TargetUsageScope = Field(..., description="Target workload evaluated")
    audit_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time of audit execution",
    )
    is_audit_passed: bool = Field(..., description="True if no blocking integrity or rights violations exist")
    total_records: int = Field(..., ge=0)
    passed_records: int = Field(..., ge=0)
    failed_records: int = Field(..., ge=0)

    # Detailed defect inventories
    missing_files: List[str] = Field(default_factory=list)
    missing_hashes: List[str] = Field(default_factory=list)
    hash_mismatches: List[Dict[str, Any]] = Field(default_factory=list)
    exact_duplicates: List[ExactDuplicateCluster] = Field(default_factory=list)
    near_duplicate_candidates: List[NearDuplicateCandidate] = Field(default_factory=list)
    conflicting_provenance: List[Dict[str, Any]] = Field(default_factory=list)
    invalid_formats: List[Dict[str, Any]] = Field(default_factory=list)
    corrupted_files: List[Dict[str, Any]] = Field(default_factory=list)
    missing_licence_information: List[str] = Field(default_factory=list)
    unapproved_records: List[str] = Field(default_factory=list)
    provenance_violations: List[ProvenanceViolation] = Field(default_factory=list)

    findings: List[AuditFinding] = Field(default_factory=list)
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def to_markdown(self) -> str:
        """Render a clean GitHub-style Markdown audit summary."""
        status_badge = "PASSED" if self.is_audit_passed else "FAILED"
        lines = [
            f"# SafeHER Dataset Audit Report: {self.dataset_name}",
            "",
            f"**Audit Status:** {status_badge}  ",
            f"**Target Workload:** `{self.target_workload.value}`  ",
            f"**Audit Date:** {self.audit_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Records Evaluated:** {self.total_records} (Passed: {self.passed_records}, Failed: {self.failed_records})  ",
            "",
            "## Summary Metrics",
            "",
            f"- **Missing Files on Disk:** {len(self.missing_files)}",
            f"- **Missing SHA-256 Digests:** {len(self.missing_hashes)}",
            f"- **Cryptographic Hash Mismatches:** {len(self.hash_mismatches)}",
            f"- **Exact Duplicate Clusters:** {len(self.exact_duplicates)}",
            f"- **Near-Duplicate Candidates Flagged:** {len(self.near_duplicate_candidates)}",
            f"- **Provenance Contradictions:** {len(self.conflicting_provenance)}",
            f"- **Corrupted / Truncated Files:** {len(self.corrupted_files)}",
            f"- **Invalid / Disallowed Formats:** {len(self.invalid_formats)}",
            f"- **Missing License Metadata:** {len(self.missing_licence_information)}",
            f"- **Unapproved Records for Workload:** {len(self.unapproved_records)}",
            f"- **Provenance Chain Inconsistencies:** {len(self.provenance_violations)}",
            "",
        ]

        if self.findings:
            lines.append("## Defect Inventory")
            lines.append("")
            lines.append("| Severity | Category | Record ID | Details | Remediation |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for f in self.findings:
                lines.append(f"| `{f.severity}` | `{f.category}` | `{f.record_id}` | {f.details} | {f.remediation} |")
            lines.append("")

        return "\n".join(lines)


class DatasetAuditor:
    """Comprehensive auditor validating dataset manifests against on-disk files and policy rules."""

    def __init__(
        self,
        duplicate_detector: Optional[DuplicateDetector] = None,
        max_image_bytes: int = 50 * 1024 * 1024,
    ):
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.max_image_bytes = max_image_bytes

    def audit_dataset(
        self,
        records: List[DatasetImageRecord],
        dataset_name: str = "SafeHER Dataset",
        manifest_id: Optional[str] = None,
        root_dir: Optional[Union[str, Path]] = None,
        target_workload: TargetUsageScope = TargetUsageScope.PRODUCTION_TRAINING,
        check_filesystem: bool = True,
        compute_perceptual_hashes: bool = True,
    ) -> DatasetAuditReport:
        """Execute complete audit across records and local filesystem assets."""
        missing_files: List[str] = []
        missing_hashes: List[str] = []
        hash_mismatches: List[Dict[str, Any]] = []
        invalid_formats: List[Dict[str, Any]] = []
        corrupted_files: List[Dict[str, Any]] = []
        missing_licenses: List[str] = []
        unapproved_records: List[str] = []
        findings: List[AuditFinding] = []
        failed_records_set: Set[str] = set()

        is_commercial = target_workload in (
            TargetUsageScope.COMMERCIAL_TRAINING,
            TargetUsageScope.PRODUCTION_TRAINING,
        )

        perceptual_hashes: Dict[str, str] = {}

        # 1. Individual record checks
        for rec in records:
            rec_id = rec.record_id
            rec_has_error = False

            # Check 1: Missing hash
            if not rec.sha256 or len(rec.sha256) != 64:
                missing_hashes.append(rec_id)
                rec_has_error = True
                findings.append(
                    AuditFinding(
                        category="MISSING_HASH",
                        record_id=rec_id,
                        severity="BLOCKER",
                        details=f"Record '{rec_id}' has missing or invalid SHA-256 hash: '{rec.sha256}'.",
                        remediation="Calculate and store 64-character lowercase hexadecimal SHA-256.",
                    )
                )

            # Check 2: Missing or unknown license
            lic_clean = (rec.original_license or "").strip().lower()
            if not lic_clean or lic_clean in ("unknown", "unspecified", "none"):
                missing_licenses.append(rec_id)
                severity = "BLOCKER" if is_commercial else "WARNING"
                if is_commercial:
                    rec_has_error = True
                findings.append(
                    AuditFinding(
                        category="MISSING_LICENSE",
                        record_id=rec_id,
                        severity=severity,
                        details=f"Record '{rec_id}' lacks verified license information: '{rec.original_license}'.",
                        remediation="Document original upstream copyright and licensing terms.",
                    )
                )

            # Check 3: Unapproved record for workload
            if is_commercial:
                if rec.rights_status not in (
                    RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
                    RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
                ):
                    unapproved_records.append(rec_id)
                    rec_has_error = True
                    findings.append(
                        AuditFinding(
                            category="UNAPPROVED_RECORD",
                            record_id=rec_id,
                            severity="BLOCKER",
                            details=(
                                f"Record '{rec_id}' has rights status '{rec.rights_status.value}'. "
                                "Commercial training permits only 'APPROVED_FOR_COMMERCIAL_TRAINING'."
                            ),
                            remediation="Obtain formal legal clearance or segregate asset to research splits.",
                        )
                    )

            # Check 4: Filesystem integrity & decode (if root_dir is provided and file exists)
            if check_filesystem and root_dir is not None:
                raw_ref = rec.image_path_or_ref
                # Only check local filesystem paths (skip non-local URIs like https://)
                if not raw_ref.startswith(("https://", "http://", "s3://", "gs://")):
                    try:
                        resolved_path = resolve_safe_relative_path(root_dir, raw_ref)
                        if not resolved_path.exists():
                            missing_files.append(rec_id)
                            rec_has_error = True
                            findings.append(
                                AuditFinding(
                                    category="MISSING_FILE",
                                    record_id=rec_id,
                                    severity="BLOCKER",
                                    details=f"Image file does not exist on disk: {resolved_path}",
                                    remediation="Restore asset to dataset repository or update path reference.",
                                )
                            )
                        else:
                            # File exists: verify hash and decode
                            actual_hash = compute_file_sha256(resolved_path)
                            if rec.sha256 and actual_hash != rec.sha256.lower():
                                mismatch_info = {
                                    "record_id": rec_id,
                                    "path": str(resolved_path),
                                    "manifest_sha256": rec.sha256,
                                    "actual_sha256": actual_hash,
                                }
                                hash_mismatches.append(mismatch_info)
                                rec_has_error = True
                                findings.append(
                                    AuditFinding(
                                        category="HASH_MISMATCH",
                                        record_id=rec_id,
                                        severity="BLOCKER",
                                        details=(
                                            f"Actual file hash ({actual_hash[:12]}...) does not match "
                                            f"manifest hash ({rec.sha256[:12]}...)."
                                        ),
                                        remediation="Investigate file tampering or update manifest with verified digest.",
                                    )
                                )

                            # Verify image format and decode
                            try:
                                meta = validate_image_file(resolved_path, max_bytes=self.max_image_bytes)
                                if compute_perceptual_hashes:
                                    try:
                                        p_hash = compute_dhash(resolved_path.read_bytes())
                                        perceptual_hashes[rec_id] = p_hash
                                    except Exception:
                                        pass
                            except InvalidFormatError as fmt_err:
                                invalid_formats.append({"record_id": rec_id, "error": str(fmt_err)})
                                rec_has_error = True
                                findings.append(
                                    AuditFinding(
                                        category="INVALID_FORMAT",
                                        record_id=rec_id,
                                        severity="BLOCKER",
                                        details=str(fmt_err),
                                        remediation="Convert asset to standard approved format (JPEG/PNG/WEBP).",
                                    )
                                )
                            except CorruptedFileError as corp_err:
                                corrupted_files.append({"record_id": rec_id, "error": str(corp_err)})
                                rec_has_error = True
                                findings.append(
                                    AuditFinding(
                                        category="CORRUPTED_FILE",
                                        record_id=rec_id,
                                        severity="BLOCKER",
                                        details=str(corp_err),
                                        remediation="Re-encode or replace corrupted image asset.",
                                    )
                                )
                            except FileIntegrityError as file_err:
                                corrupted_files.append({"record_id": rec_id, "error": str(file_err)})
                                rec_has_error = True
                                findings.append(
                                    AuditFinding(
                                        category="FILE_INTEGRITY_VIOLATION",
                                        record_id=rec_id,
                                        severity="BLOCKER",
                                        details=str(file_err),
                                        remediation="Verify file size and non-empty status.",
                                    )
                                )

                    except PathTraversalError as trav_err:
                        corrupted_files.append({"record_id": rec_id, "error": str(trav_err)})
                        rec_has_error = True
                        findings.append(
                            AuditFinding(
                                category="PATH_TRAVERSAL_ATTEMPT",
                                record_id=rec_id,
                                severity="BLOCKER",
                                details=str(trav_err),
                                remediation="Purge insecure relative path references containing '..' or root escapes.",
                            )
                        )

            if rec_has_error:
                failed_records_set.add(rec_id)

        # 2. Duplicate detection (Exact + Near)
        dup_report = self.duplicate_detector.analyze_dataset(
            records=records,
            cached_hashes=perceptual_hashes if perceptual_hashes else None,
        )

        conflicting_provenance_list: List[Dict[str, Any]] = []
        for cluster in dup_report.exact_duplicate_clusters:
            if cluster.has_provenance_conflict:
                failed_records_set.update(cluster.record_ids)
                conflict_item = {
                    "sha256": cluster.sha256,
                    "record_ids": cluster.record_ids,
                    "conflicting_fields": cluster.conflicting_fields,
                }
                conflicting_provenance_list.append(conflict_item)
                findings.append(
                    AuditFinding(
                        category="CONFLICTING_PROVENANCE",
                        record_id=cluster.record_ids[0],
                        severity="BLOCKER",
                        details=(
                            f"Identical image hash ({cluster.sha256[:12]}...) appears across {cluster.count} records "
                            f"with conflicting metadata in fields: {cluster.conflicting_fields}."
                        ),
                        remediation="Resolve contradictory metadata and eliminate duplicate records.",
                    )
                )

        # Near-duplicate candidates
        for near_cand in dup_report.near_duplicate_candidates:
            findings.append(
                AuditFinding(
                    category="NEAR_DUPLICATE_CANDIDATE",
                    record_id=near_cand.record_id_a,
                    severity="INFO",
                    details=(
                        f"Potential visual near-duplicate detected with record '{near_cand.record_id_b}' "
                        f"(Hamming distance: {near_cand.hamming_distance}/64, score: {near_cand.similarity_score}). {near_cand.notes}"
                    ),
                    remediation="Conduct manual forensic review. Note: Perceptual similarity is NOT biometric identity.",
                )
            )

        # 3. Provenance chain consistency check
        provenance_violations = verify_provenance_consistency(records)
        for p_violation in provenance_violations:
            failed_records_set.add(p_violation.record_id)
            findings.append(
                AuditFinding(
                    category=p_violation.violation_type.value,
                    record_id=p_violation.record_id,
                    severity="BLOCKER",
                    details=p_violation.message,
                    remediation=p_violation.actionable_fix,
                )
            )

        total_cnt = len(records)
        failed_cnt = len(failed_records_set)
        passed_cnt = total_cnt - failed_cnt
        has_blockers = any(f.severity == "BLOCKER" for f in findings)
        is_passed = not has_blockers

        summary = {
            "total_records": total_cnt,
            "passed_records": passed_cnt,
            "failed_records": failed_cnt,
            "missing_files_count": len(missing_files),
            "missing_hashes_count": len(missing_hashes),
            "hash_mismatches_count": len(hash_mismatches),
            "exact_duplicate_clusters_count": len(dup_report.exact_duplicate_clusters),
            "near_duplicate_candidates_count": len(dup_report.near_duplicate_candidates),
            "conflicting_provenance_count": len(conflicting_provenance_list),
            "invalid_formats_count": len(invalid_formats),
            "corrupted_files_count": len(corrupted_files),
            "missing_licenses_count": len(missing_licenses),
            "unapproved_records_count": len(unapproved_records),
            "provenance_violations_count": len(provenance_violations),
            "total_findings_count": len(findings),
        }

        return DatasetAuditReport(
            dataset_name=dataset_name,
            manifest_id=manifest_id,
            target_workload=target_workload,
            is_audit_passed=is_passed,
            total_records=total_cnt,
            passed_records=passed_cnt,
            failed_records=failed_cnt,
            missing_files=missing_files,
            missing_hashes=missing_hashes,
            hash_mismatches=hash_mismatches,
            exact_duplicates=dup_report.exact_duplicate_clusters,
            near_duplicate_candidates=dup_report.near_duplicate_candidates,
            conflicting_provenance=conflicting_provenance_list,
            invalid_formats=invalid_formats,
            corrupted_files=corrupted_files,
            missing_licence_information=missing_licenses,
            unapproved_records=unapproved_records,
            provenance_violations=provenance_violations,
            findings=findings,
            summary_metrics=summary,
        )

    def audit_manifest(
        self,
        manifest: DatasetManifest,
        root_dir: Optional[Union[str, Path]] = None,
        target_workload: Optional[TargetUsageScope] = None,
        check_filesystem: bool = True,
        compute_perceptual_hashes: bool = True,
    ) -> DatasetAuditReport:
        """Audit an entire DatasetManifest instance against local storage and compliance policies."""
        target = target_workload or manifest.target_use
        return self.audit_dataset(
            records=manifest.records,
            dataset_name=manifest.dataset_name,
            manifest_id=manifest.manifest_id,
            root_dir=root_dir,
            target_workload=target,
            check_filesystem=check_filesystem,
            compute_perceptual_hashes=compute_perceptual_hashes,
        )
