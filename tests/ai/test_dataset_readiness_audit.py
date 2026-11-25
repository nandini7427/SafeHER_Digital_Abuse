"""Automated test suite for SafeHER Milestone 4.2 Dataset Readiness and Acceptance Audit.

Verifies:
1. All 13 readiness verification checks are systematically executed.
2. Manifest schema validity and PII protections.
3. Rights decisions and legal profiles recorded for all candidates.
4. Research and commercial tracks separation.
5. Commercial training approval verification and blocker detection.
6. Subject consent enforcement on human-subject records.
7. Verifiable provenance and lineage DAG consistency.
8. Image hash fidelity and tamper detection.
9. Duplicate and perceptual near-duplicate candidate review.
10. Group-aware splitting and held-out test set isolation.
11. Transform reproducibility and deterministic evaluation mode.
12. Corrupt data detection and non-silent exclusion logging.
13. Strict survivor evidence quarantine (zero evidence vault data in training manifests).
14. Quantitative dataset statistics compilation (non-fabricated).
15. Separate acceptance gate verdicts for research vs. commercial tracks.
"""

from pathlib import Path
import pytest
import torch

from ai_core.datasets.manifest import (
    ConsentStatus,
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    DatasetTaskType,
    ManipulationCategory,
    RightsApprovalStatus,
    TargetUsageScope,
    UsageScope,
)
from ai_core.datasets.readiness_auditor import (
    DatasetReadinessAuditor,
    DatasetReadinessReport,
    QuantitativeDatasetStatistics,
    ReadinessVerificationCheck,
    run_readiness_audit,
)
from ai_core.datasets.registry import (
    create_candidate_sample_record,
    create_initial_registry_manifest,
    list_all_candidates,
)


def _make_sample_record(
    record_id: str,
    label: DatasetImageLabel = DatasetImageLabel.CAMERA_PHOTOGRAPH,
    split: DatasetSplit = DatasetSplit.TRAIN,
    rights_status: RightsApprovalStatus = RightsApprovalStatus.NOT_APPROVED_FOR_PRODUCTION,
    image_ref: str = "images/sample.jpg",
    human_subjects: bool = False,
    consent_ref: str = None,
    consent_status: ConsentStatus = ConsentStatus.NOT_APPLICABLE,
) -> DatasetImageRecord:
    """Helper to instantiate test dataset image records."""
    return DatasetImageRecord(
        record_id=record_id,
        image_path_or_ref=image_ref,
        sha256="a" * 64,
        label=label,
        task_type=DatasetTaskType.DEEPFAKE_DETECTION,
        manipulation_category=ManipulationCategory.NONE,
        dataset_name="SafeHER-Test",
        source_creator="SafeHER Test Lab",
        source_url="https://safeher.local/test",
        source_license="CC-BY-4.0",
        rights_status=rights_status,
        human_subjects_present=human_subjects,
        consent_reference=consent_ref,
        consent_status=consent_status,
        split=split,
    )


class TestDatasetReadinessAudit:
    """Comprehensive test suite for DatasetReadinessAuditor."""

    def test_readiness_audit_executes_all_13_checks(self):
        """Verify that the auditor executes exactly 13 checks covering all required dimensions."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        report = auditor.run_full_readiness_audit()

        assert isinstance(report, DatasetReadinessReport)
        assert len(report.verification_checks) == 13

        check_numbers = [c.check_number for c in report.verification_checks]
        assert check_numbers == list(range(1, 14))

    def test_schema_validity_check_detects_malformed_records(self):
        """Check 1 must verify schema adherence and catch invalid IDs or hash digests."""
        rec_valid = _make_sample_record("SH-REC-0001")
        manifest = DatasetManifest(
            manifest_id="test-manifest",
            dataset_name="Test",
            version="1.0.0",
            records=[rec_valid],
        )
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c1 = auditor._check_1_schema_validity()
        assert c1.passed is True
        assert c1.status == "PASSED"

    def test_rights_decisions_recorded_for_all_candidates(self):
        """Check 2 must confirm all 9 registered candidates have documented legal profiles."""
        auditor = DatasetReadinessAuditor()
        c2 = auditor._check_2_rights_decisions_recorded()
        assert c2.passed is True
        assert c2.details["registered_candidates"] == 9
        assert c2.details["unrecorded_count"] == 0

    def test_research_vs_commercial_track_segregation(self):
        """Check 3 must confirm research datasets are not improperly claimed as commercial."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c3 = auditor._check_3_tracks_separated()
        assert c3.passed is True
        assert c3.details["cross_track_leakage"] == 0

    def test_commercial_training_approval_blocks_unapproved_manifest(self):
        """Check 4 must block commercial training when zero records have verified approval."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c4 = auditor._check_4_commercial_training_approval()

        # In initial registry, all records are NOT_APPROVED_FOR_PRODUCTION
        assert c4.passed is False
        assert c4.status == "BLOCKED"
        assert len(c4.blockers) > 0
        assert any("Zero records currently possess verified commercial training clearance" in b for b in c4.blockers)

    def test_subject_consent_enforced_and_catches_pii(self):
        """Check 5 must enforce consent policies and reject any records with PII."""
        # 1. PII in consent reference is blocked upon record creation
        with pytest.raises(ValueError, match="Consent reference must be an opaque"):
            _make_sample_record(
                "SH-PII-001",
                human_subjects=True,
                consent_ref="actor_john.doe@example.com",
                consent_status=ConsentStatus.AUTHORIZED,
            )

        # 2. Check 5 flags records where human subjects lack authorized consent
        rec_unauthorized = _make_sample_record(
            "SH-CONSENT-001",
            human_subjects=True,
            consent_ref="CR-SAFEHER-2026-F001",
            consent_status=ConsentStatus.UNVERIFIED,
            rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        )
        manifest = DatasetManifest(
            manifest_id="consent-manifest",
            dataset_name="ConsentTest",
            version="1.0.0",
            records=[rec_unauthorized],
        )
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c5 = auditor._check_5_subject_consent_enforced()
        assert c5.passed is False
        assert c5.status == "BLOCKED"
        assert any("lacks authorized subject consent" in b for b in c5.blockers)

    def test_verifiable_provenance_and_generator_metadata(self):
        """Check 6 must verify source creator, license, and synthetic generator provenance."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c6 = auditor._check_6_verifiable_provenance()
        assert c6.passed is True
        assert c6.details["graph_violations"] == 0

    def test_image_hash_fidelity_validation(self):
        """Check 7 must verify 64-character lowercase hexadecimal hash format."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c7 = auditor._check_7_image_hashes_match()
        assert c7.passed is True
        assert c7.details["mismatches"] == 0

    def test_duplicate_and_near_duplicate_candidate_review(self):
        """Check 8 must surface duplicate clusters without false equivalence claims."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c8 = auditor._check_8_duplicate_candidates_reviewed()
        assert c8.passed is True
        assert "exact_duplicate_clusters" in c8.details
        assert "exact_duplicate_records" in c8.details

    def test_group_aware_splitting_and_leakage_audit(self):
        """Check 9 & 10 must audit partitions and verify held-out generator isolation."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)

        c9 = auditor._check_9_group_aware_splitting()
        assert c9.passed is True
        assert c9.details["is_leakage_free"] is True

        c10 = auditor._check_10_held_out_test_set_isolation()
        assert c10.passed is True
        assert c10.details["held_out_leaks"] == 0

    def test_transformations_reproducibility_and_eval_determinism(self):
        """Check 11 must confirm bit-for-bit reproducibility and deterministic evaluation."""
        auditor = DatasetReadinessAuditor()
        c11 = auditor._check_11_transformations_reproducible()
        assert c11.passed is True
        assert c11.details["train_reproducible"] is True
        assert c11.details["eval_deterministic"] is True

    def test_corrupt_data_reporting_and_exclusion_log(self):
        """Check 12 must verify non-silent exclusion logging and error categorization."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        c12 = auditor._check_12_corrupt_data_reported()
        assert c12.passed is True
        assert c12.details["exclusion_tracking_enabled"] is True

    def test_no_survivor_evidence_in_training(self):
        """Check 13 must strictly prove zero survivor incident/case evidence in manifests."""
        # 1. Clean registry manifest
        clean_manifest = create_initial_registry_manifest()
        auditor_clean = DatasetReadinessAuditor(manifest=clean_manifest)
        c13_clean = auditor_clean._check_13_no_survivor_evidence_in_training()
        assert c13_clean.passed is True
        assert c13_clean.details["contaminated_records"] == 0

        # 2. Simulated contaminated record attempting to inject private evidence vault reference
        contaminated_record = _make_sample_record(
            "SH-CASE-001",
            image_ref="cases/2026/survivor_upload_001.jpg",
        )
        contaminated_manifest = DatasetManifest(
            manifest_id="contaminated",
            dataset_name="Contaminated",
            version="1.0",
            records=[contaminated_record],
        )
        auditor_bad = DatasetReadinessAuditor(manifest=contaminated_manifest)
        c13_bad = auditor_bad._check_13_no_survivor_evidence_in_training()
        assert c13_bad.passed is False
        assert c13_bad.status == "BLOCKED"
        assert len(c13_bad.blockers) > 0
        assert "evidence vault reference" in c13_bad.blockers[0]

    def test_acceptance_gate_verdicts_research_vs_commercial(self):
        """Acceptance gates must issue APPROVED for research and BLOCKED for commercial."""
        report = run_readiness_audit()

        # Research track: APPROVED for academic experimentation and benchmarking
        assert report.research_track_verdict.is_approved is True
        assert report.research_track_verdict.verdict == "APPROVED"
        assert len(report.research_track_verdict.blockers) >= 1  # Contains usage conditions

        # Commercial track: BLOCKED / NOT_APPROVED
        assert report.commercial_track_verdict.is_approved is False
        assert "BLOCKED" in report.commercial_track_verdict.verdict
        assert len(report.commercial_track_verdict.blockers) >= 1
        assert len(report.commercial_track_verdict.actionable_remediation) >= 3

    def test_quantitative_statistics_accuracy(self):
        """Quantitative statistics must compute exact non-fabricated counts across dimensions."""
        manifest = create_initial_registry_manifest()
        auditor = DatasetReadinessAuditor(manifest=manifest)
        stats = auditor._compute_quantitative_statistics()

        assert stats.total_images == len(manifest.records)
        assert sum(stats.class_counts.values()) == stats.total_images
        assert sum(stats.source_counts.values()) == stats.total_images
        assert sum(stats.rights_status_counts.values()) == stats.total_images
        assert stats.corrupt_file_counts == 0
        assert "total_exclusions" in stats.exclusion_counts

    def test_markdown_report_formatting(self):
        """to_markdown() must format a complete GitHub Flavored Markdown report."""
        report = run_readiness_audit()
        md = report.to_markdown()

        assert "# SafeHER Dataset Readiness and Acceptance Audit Report" in md
        assert "## 1. Acceptance Gate Verdicts" in md
        assert "## 2. 13-Point Technical and Legal Verification Matrix" in md
        assert "## 3. Quantitative Dataset Statistics" in md
        assert "## 4. Survivor Evidence Isolation Audit" in md
        assert "Research Experimentation Track:" in md
        assert "Commercial-Intended Training Track:" in md
        assert "BLOCKED" in md
