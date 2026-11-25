"""Rights validation engine for SafeHER image dataset pipeline.

Enforces:
1. Rejection of unknown or unverified licenses for commercial training.
2. Flagging of non-commercial datasets as research-only.
3. Individual source-license verification for aggregated datasets (umbrella declarations cannot override upstream restrictions).
4. Attribution metadata enforcement for attribution-mandating licenses.
5. Explicit human-subject consent authorization for face-based commercial training data.
6. Commercial-use verification for synthetic-image generator outputs.
7. Strict isolation preventing research-only images from entering production-training manifests.
8. Detection of duplicate image hashes and conflicting provenance records.

DISCLAIMER: This validator reports technical rights review status and unresolved
approval requirements. It does not constitute formal legal certification or legal advice.
"""

from collections import defaultdict
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.manifest import (
    CommercialUseStatus,
    ConsentStatus,
    DatasetImageRecord,
    DatasetManifest,
    ManipulationCategory,
    RightsApprovalStatus,
    TargetUsageScope,
    UsageScope,
)


class ValidationSeverity(str, Enum):
    """Severity of a validation finding."""

    BLOCKER = "BLOCKER"
    WARNING = "WARNING"
    INFO = "INFO"


class RightsViolation(BaseModel):
    """Structured report of an unresolved rights requirement or policy violation."""

    rule_code: str = Field(..., description="Machine-readable rule identifier (e.g. 'ERR_UNKNOWN_LICENSE')")
    severity: ValidationSeverity = Field(..., description="Violation severity level")
    message: str = Field(..., description="Human-readable explanation of violation")
    record_id: Optional[str] = Field(None, description="Affected record ID, if applicable")
    unresolved_requirement: Optional[str] = Field(None, description="Actionable approval requirement")

    model_config = ConfigDict(frozen=True)


class RightsReviewStatus(str, Enum):
    """Categorical status of technical rights review."""

    APPROVED_FOR_TARGET = "APPROVED_FOR_TARGET"
    REJECTED = "REJECTED"
    ACTION_REQUIRED_UNRESOLVED = "ACTION_REQUIRED_UNRESOLVED"


class ValidationReport(BaseModel):
    """Comprehensive rights validation and technical provenance report."""

    is_valid: bool = Field(..., description="True if no blocking violations exist for target usage")
    status: RightsReviewStatus = Field(..., description="Overall review status")
    target_usage: TargetUsageScope = Field(..., description="Evaluated workload scope")
    total_records: int = Field(..., ge=0, description="Total image records evaluated")
    passed_records: int = Field(..., ge=0, description="Records without blocking errors")
    failed_records: int = Field(..., ge=0, description="Records with one or more blocking errors")
    violations: List[RightsViolation] = Field(default_factory=list, description="All violations found")
    unresolved_approval_requirements: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of unresolved requirements",
    )
    duplicate_hashes_found: List[str] = Field(
        default_factory=list,
        description="List of SHA-256 hashes appearing more than once",
    )
    conflicting_provenance_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Records with identical hash but contradictory provenance",
    )
    technical_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Technical metrics and breakdown of reviewed manifest",
    )
    legal_disclaimer: str = Field(
        default=(
            "SafeHER Technical Rights Engine: This report conveys technical verification of metadata, "
            "cryptographic hashes, license attributes, and policy compliance. It does not constitute "
            "formal legal certification or legal advice."
        ),
        description="Mandatory legal disclaimer",
    )

    model_config = ConfigDict(frozen=True)


# License classification taxonomies
_PERMISSIVE_COMMERCIAL_LICENSES: Set[str] = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "unlicense",
    "safeher-first-party-commercial",
    "proprietary-commercial-authorized",
    "open-data-commons-pddl",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-2.0",
}

_NON_COMMERCIAL_LICENSE_TOKENS: Set[str] = {
    "-nc",
    "nc-",
    "non-commercial",
    "noncommercial",
    "academic",
    "research-only",
    "personal-use",
}

_KNOWN_NON_COMMERCIAL_LICENSES: Set[str] = {
    "cc-by-nc-4.0",
    "cc-by-nc-3.0",
    "cc-by-nc-2.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-nd-4.0",
    "academic-research-only",
    "custom-academic",
    "faceforensics-academic-agreement",
    "deeperforensics-academic-license",
    "genimage-academic-license",
    "dfdc-challenge-agreement",
    "eddc-2026-academic",
}

_ATTRIBUTION_MANDATED_LICENSES: Set[str] = {
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-2.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-nc-4.0",
    "cc-by-nc-3.0",
    "cc-by-nc-sa-4.0",
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
}

# Known verified in-house generators with pre-cleared commercial training outputs
_KNOWN_COMMERCIAL_GENERATORS: Set[str] = {
    "safeher-procedural-v1",
    "safeher-deterministic-forensics-v1",
    "safeher-synthetic-camera-v1",
    "adobe-firefly-v2",
}

_KNOWN_ACADEMIC_OR_RESEARCH_GENERATORS: Set[str] = {
    "stablediffusion-v1.4",
    "stablediffusion-v1.5",
    "stablediffusion-v2.1",
    "stablediffusion-xl",
    "midjourney-v5",
    "midjourney-v6",
    "dall-e-2",
    "dall-e-3",
    "faceswap-gan",
    "deepfakes",
    "face2face",
    "neuraltextures",
    "stylegan",
    "stylegan2",
    "stylegan3",
    "wukong",
    "glide",
    "vqdm",
    "adm",
    "biggan",
}

_FACE_MANIPULATION_CATEGORIES: Set[ManipulationCategory] = {
    ManipulationCategory.DEEPFAKE_FACE_SWAP,
    ManipulationCategory.FACE_REENACTMENT,
}


class RightsValidationEngine:
    """Engine executing cryptographic, licensing, consent, and provenance checks on dataset records."""

    def __init__(
        self,
        known_commercial_licenses: Optional[Set[str]] = None,
        known_commercial_generators: Optional[Set[str]] = None,
    ):
        self.commercial_licenses = set(known_commercial_licenses or _PERMISSIVE_COMMERCIAL_LICENSES)
        self.commercial_generators = set(known_commercial_generators or _KNOWN_COMMERCIAL_GENERATORS)

    def is_known_commercial_license(self, license_id: Optional[str]) -> bool:
        """Check if license identifier is a recognized permissive commercial license."""
        if not license_id:
            return False
        clean = license_id.strip().lower()
        return clean in self.commercial_licenses

    def is_non_commercial_license(self, license_id: Optional[str]) -> bool:
        """Determine if license has non-commercial, academic, or research-only restrictions."""
        if not license_id:
            return False
        clean = license_id.strip().lower()
        if clean in _KNOWN_NON_COMMERCIAL_LICENSES:
            return True
        return any(token in clean for token in _NON_COMMERCIAL_LICENSE_TOKENS)

    def requires_attribution(self, license_id: Optional[str]) -> bool:
        """Check if license mandates public attribution notice."""
        if not license_id:
            return False
        clean = license_id.strip().lower()
        return clean in _ATTRIBUTION_MANDATED_LICENSES

    def validate_record(
        self,
        record: DatasetImageRecord,
        target_usage: TargetUsageScope = TargetUsageScope.PRODUCTION_TRAINING,
        is_aggregated: bool = False,
        general_license: Optional[str] = None,
    ) -> List[RightsViolation]:
        """Validate an individual image record against specified target usage and policy constraints."""
        violations: List[RightsViolation] = []
        is_commercial_target = target_usage in (
            TargetUsageScope.COMMERCIAL_TRAINING,
            TargetUsageScope.PRODUCTION_TRAINING,
        )

        # ----------------------------------------------------------------------
        # Check: Explicit Rejection and Consent Revocation
        # ----------------------------------------------------------------------
        if record.rights_status == RightsApprovalStatus.REJECTED:
            violations.append(
                RightsViolation(
                    rule_code="ERR_REJECTED_RECORD",
                    severity=ValidationSeverity.BLOCKER,
                    message=(
                        f"Record '{record.record_id}' is explicitly marked REJECTED. "
                        "Rejected records are barred from all model training, research, and evaluation workflows."
                    ),
                    record_id=record.record_id,
                    unresolved_requirement="Purge rejected record from manifest or re-review provenance.",
                )
            )

        if record.consent_status == ConsentStatus.REVOKED:
            violations.append(
                RightsViolation(
                    rule_code="ERR_REVOKED_CONSENT",
                    severity=ValidationSeverity.BLOCKER,
                    message=(
                        f"Record '{record.record_id}' has REVOKED human-subject consent. "
                        "All processing, training, and evaluation must immediately cease under data subject privacy rights."
                    ),
                    record_id=record.record_id,
                    unresolved_requirement="Purge revoked asset immediately from all datasets and training splits.",
                )
            )

        # ----------------------------------------------------------------------
        # Check: Cryptographic SHA-256 Digest Presence and Format
        # ----------------------------------------------------------------------
        if not record.sha256 or len(record.sha256) != 64 or not re.match(r"^[0-9a-f]{64}$", record.sha256):
            violations.append(
                RightsViolation(
                    rule_code="ERR_MISSING_OR_INVALID_HASH",
                    severity=ValidationSeverity.BLOCKER,
                    message=f"Missing or malformed SHA-256 hash for record '{record.record_id}'.",
                    record_id=record.record_id,
                    unresolved_requirement="Compute and record valid 64-character lowercase SHA-256 digest.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 1: Reject unknown licenses for commercial training
        # ----------------------------------------------------------------------
        lic_clean = (record.original_license or "").strip().lower()
        if is_commercial_target:
            if not lic_clean or lic_clean in ("unknown", "unspecified", "none", "custom"):
                violations.append(
                    RightsViolation(
                        rule_code="ERR_UNKNOWN_LICENSE",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' has unknown or unspecified license: '{record.original_license}'. "
                            "Commercial training strictly prohibits unverified or unknown licenses."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Identify, document, and verify original upstream license terms.",
                    )
                )
            elif not self.is_known_commercial_license(lic_clean):
                if not self.is_non_commercial_license(lic_clean):
                    violations.append(
                        RightsViolation(
                            rule_code="ERR_UNKNOWN_LICENSE",
                            severity=ValidationSeverity.BLOCKER,
                            message=(
                                f"License '{record.original_license}' is not in the certified commercial license registry. "
                                "Requires formal technical and legal clearance."
                            ),
                            record_id=record.record_id,
                            unresolved_requirement=f"Submit license '{record.original_license}' for commercial clearance review.",
                        )
                    )

        # ----------------------------------------------------------------------
        # Rule 2: Flag non-commercial datasets as research-only
        # ----------------------------------------------------------------------
        if self.is_non_commercial_license(lic_clean):
            # Check if record claims commercial training in allowed usage
            if UsageScope.COMMERCIAL_TRAINING in record.allowed_usage:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_NON_COMMERCIAL_RESTRICTION",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' has non-commercial license '{record.original_license}', "
                            "but declares 'commercial_training' in allowed_usage. Non-commercial datasets must be flagged research-only."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Restrict allowed_usage to research and evaluation only.",
                    )
                )
            # Commercial target violation
            if is_commercial_target:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_NON_COMMERCIAL_RESTRICTION",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Non-commercial asset '{record.record_id}' (license: '{record.original_license}') "
                            "is strictly prohibited from commercial or production training workloads."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Exclude non-commercial asset from commercial training manifests.",
                    )
                )

        # ----------------------------------------------------------------------
        # Rule 3: Aggregated datasets require individual upstream source verification
        # ----------------------------------------------------------------------
        record_is_agg = record.is_aggregated or is_aggregated
        if record_is_agg:
            if not record.original_source_ref or not record.original_source_ref.strip():
                violations.append(
                    RightsViolation(
                        rule_code="ERR_AGGREGATE_SOURCE_UNVERIFIED",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Aggregated record '{record.record_id}' from dataset '{record.source_dataset}' "
                            "is missing individual upstream source reference (original_source_ref)."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Record verifiable upstream URL or original source repository reference.",
                    )
                )
            # Upstream restrictions cannot be overridden by general license
            if general_license:
                gen_clean = general_license.strip().lower()
                if self.is_known_commercial_license(gen_clean) and self.is_non_commercial_license(lic_clean):
                    violations.append(
                        RightsViolation(
                            rule_code="ERR_AGGREGATE_UPSTREAM_RESTRICTION_CONFLICT",
                            severity=ValidationSeverity.BLOCKER,
                            message=(
                                f"General dataset license '{general_license}' cannot override upstream restriction "
                                f"'{record.original_license}' for record '{record.record_id}'. Upstream restrictions take precedence."
                            ),
                            record_id=record.record_id,
                            unresolved_requirement="Enforce upstream non-commercial restriction and segregate asset from commercial pipelines.",
                        )
                    )

        # ----------------------------------------------------------------------
        # Rule 4: Require attribution metadata when applicable
        # ----------------------------------------------------------------------
        attribution_needed = (
            self.requires_attribution(lic_clean)
            or (record.attribution is not None and record.attribution.required)
            or bool(record.attribution_requirements)
        )
        if attribution_needed:
            has_attribution = (
                (record.attribution is not None and bool(record.attribution.attribution_text))
                or (bool(record.attribution_requirements) and bool(record.source_creator))
            )
            if not has_attribution:
                severity = ValidationSeverity.BLOCKER if is_commercial_target else ValidationSeverity.WARNING
                violations.append(
                    RightsViolation(
                        rule_code="ERR_MISSING_ATTRIBUTION",
                        severity=severity,
                        message=(
                            f"License '{record.original_license}' requires attribution for record '{record.record_id}', "
                            "but no valid attribution notice or creator metadata was provided."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Provide creator attribution text and upstream license notice.",
                    )
                )

        # ----------------------------------------------------------------------
        # Rule 5: Require explicit human-subject consent for face-based commercial training
        # ----------------------------------------------------------------------
        faces_present = record.human_subjects_present or record.manipulation_category in _FACE_MANIPULATION_CATEGORIES
        if faces_present and is_commercial_target:
            if not record.consent_reference or not record.consent_reference.strip():
                violations.append(
                    RightsViolation(
                        rule_code="ERR_MISSING_ACTOR_CONSENT",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Face-based commercial training record '{record.record_id}' lacks human-subject consent reference. "
                            "Commercial training on human facial likeness requires explicit authorized consent."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Obtain and record verified non-identifying commercial biometric consent reference.",
                    )
                )
            elif record.consent_status != ConsentStatus.AUTHORIZED:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_MISSING_ACTOR_CONSENT",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Face-based commercial training record '{record.record_id}' has consent status: "
                            f"'{record.consent_status.value}'. Must be 'authorized' for commercial training."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Elevate and verify consent authorization status to 'authorized'.",
                    )
                )

        # ----------------------------------------------------------------------
        # Rule 6: Commercial-use verification for synthetic generator outputs
        # ----------------------------------------------------------------------
        is_synthetic = (
            bool(record.generator_name)
            or record.manipulation_category
            in (
                ManipulationCategory.DIFFUSION_GENERATED,
                ManipulationCategory.GAN_GENERATED,
                ManipulationCategory.DIFFUSION_INPAINTING,
                ManipulationCategory.SYNTHETIC_EDIT,
                ManipulationCategory.DEEPFAKE_FACE_SWAP,
            )
        )
        if is_synthetic and is_commercial_target:
            gen_name_clean = (record.generator_name or "").strip().lower()
            gen_permitted = (
                (record.generator_info is not None and record.generator_info.commercial_use_permitted)
                or (gen_name_clean in self.commercial_generators)
            )
            if not gen_permitted:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_UNVERIFIED_GENERATOR_RIGHTS",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Synthetic generator '{record.generator_name}' outputs have not been verified for commercial use. "
                            "Commercial model training requires explicit generator terms authorization."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement=(
                            f"Verify upstream commercial license or terms of service for generator '{record.generator_name}'."
                        ),
                    )
                )

        # ----------------------------------------------------------------------
        # Rule 7: Prevent research-only images from entering commercial training
        # ----------------------------------------------------------------------
        # Detect contradictory internal terms
        if record.commercial_use_status == CommercialUseStatus.PROHIBITED:
            has_commercial_claim = (
                UsageScope.COMMERCIAL_TRAINING in record.allowed_usage
                or getattr(record, "training_eligibility", False)
                or record.rights_status in (
                    RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
                    RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
                )
            )
            if has_commercial_claim:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_CONFLICTING_RESTRICTIONS_PENDING_REVIEW",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' has contradictory permissions: "
                            "commercial_use_status is 'PROHIBITED', but commercial rights/usage are declared."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Resolve internal provenance conflict before submitting record for review.",
                    )
                )

        if is_commercial_target:
            if record.rights_status == RightsApprovalStatus.EXPIRED_OR_REQUIRES_REVIEW:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_EXPIRED_OR_REQUIRES_REVIEW",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' has expired licensing or rights review status "
                            "('EXPIRED_OR_REQUIRES_REVIEW'). Commercial training strictly requires active approval."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Conduct periodic rights re-review and renew commercial clearance.",
                    )
                )
            elif record.rights_status == RightsApprovalStatus.PENDING_REVIEW or record.commercial_use_status == CommercialUseStatus.REQUIRES_REVIEW:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_CONFLICTING_RESTRICTIONS_PENDING_REVIEW",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' has status 'PENDING_REVIEW' or 'REQUIRES_REVIEW'. "
                            "Datasets with unverified or conflicting terms are blocked from commercial training pending formal review."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Complete formal rights and license agreement review.",
                    )
                )
            elif record.rights_status not in (
                RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
                RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
            ):
                violations.append(
                    RightsViolation(
                        rule_code="ERR_RESEARCH_ONLY_IN_PRODUCTION",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' has rights status '{record.rights_status.value}'. "
                            "Only 'APPROVED_FOR_COMMERCIAL_TRAINING' records may enter commercial/production training manifests."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Complete formal rights approval checklist to certify asset for commercial training.",
                    )
                )
            if UsageScope.COMMERCIAL_TRAINING not in record.allowed_usage and not getattr(record, "training_eligibility", False):
                violations.append(
                    RightsViolation(
                        rule_code="ERR_RESEARCH_ONLY_IN_PRODUCTION",
                        severity=ValidationSeverity.BLOCKER,
                        message=(
                            f"Record '{record.record_id}' is not eligible for commercial training. "
                            "Commercial training strictly forbids research-only or unverified records."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Verify and expand allowed usage scope to commercial_training.",
                    )
                )
        else:
            # Non-commercial target: flag expired records as warning
            if record.rights_status == RightsApprovalStatus.EXPIRED_OR_REQUIRES_REVIEW:
                violations.append(
                    RightsViolation(
                        rule_code="ERR_EXPIRED_OR_REQUIRES_REVIEW",
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Record '{record.record_id}' has rights status 'EXPIRED_OR_REQUIRES_REVIEW'. "
                            "Requires re-verification before continued long-term research usage."
                        ),
                        record_id=record.record_id,
                        unresolved_requirement="Schedule periodic provenance and license re-audit.",
                    )
                )

        # ----------------------------------------------------------------------
        # Rule: Unseen generator metadata detection
        # ----------------------------------------------------------------------
        if record.generator_name:
            gen_clean = record.generator_name.strip().lower()
            all_known = self.commercial_generators | _KNOWN_ACADEMIC_OR_RESEARCH_GENERATORS
            if gen_clean not in all_known:
                if record.generator_info is None or not record.generator_info.terms_reference:
                    severity = ValidationSeverity.BLOCKER if is_commercial_target else ValidationSeverity.WARNING
                    violations.append(
                        RightsViolation(
                            rule_code="ERR_UNSEEN_GENERATOR_METADATA",
                            severity=severity,
                            message=(
                                f"Unseen generator '{record.generator_name}' lacking registered provenance and licensing terms."
                            ),
                            record_id=record.record_id,
                            unresolved_requirement=(
                                f"Register generator '{record.generator_name}' with verified terms of service and commercial clearance."
                            ),
                        )
                    )

        return violations

    def validate_manifest(
        self,
        manifest: DatasetManifest,
        target_usage: Optional[TargetUsageScope] = None,
    ) -> ValidationReport:
        """Validate an entire dataset manifest, including duplicate hash and conflicting provenance detection."""
        target = target_usage or manifest.target_use
        return self.validate_records(
            records=manifest.records,
            target_usage=target,
            is_aggregated=manifest.is_aggregated,
            general_license=manifest.general_license,
            manifest_id=manifest.manifest_id,
            dataset_name=manifest.dataset_name,
        )

    def validate_records(
        self,
        records: List[DatasetImageRecord],
        target_usage: TargetUsageScope = TargetUsageScope.PRODUCTION_TRAINING,
        is_aggregated: bool = False,
        general_license: Optional[str] = None,
        manifest_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> ValidationReport:
        """Validate a collection of image records with cross-record collision and provenance checks."""
        all_violations: List[RightsViolation] = []
        unresolved_set: Set[str] = set()

        # Group by SHA-256 for duplicate and provenance collision detection
        hash_to_records: Dict[str, List[DatasetImageRecord]] = defaultdict(list)
        failed_record_ids: Set[str] = set()

        # 1. Per-record validation
        for rec in records:
            clean_hash = (rec.sha256 or "").strip().lower()
            if clean_hash:
                hash_to_records[clean_hash].append(rec)

            rec_violations = self.validate_record(
                record=rec,
                target_usage=target_usage,
                is_aggregated=is_aggregated,
                general_license=general_license,
            )
            if rec_violations:
                all_violations.extend(rec_violations)
                has_blocker = any(v.severity == ValidationSeverity.BLOCKER for v in rec_violations)
                if has_blocker:
                    failed_record_ids.add(rec.record_id)
                for v in rec_violations:
                    if v.unresolved_requirement:
                        unresolved_set.add(v.unresolved_requirement)

        # ----------------------------------------------------------------------
        # Rule 8: Detect duplicate image hashes and conflicting provenance records
        # ----------------------------------------------------------------------
        duplicate_hashes: List[str] = []
        conflicting_provenance: List[Dict[str, Any]] = []

        for h, group in hash_to_records.items():
            if len(group) > 1:
                duplicate_hashes.append(h)
                rec_ids = [r.record_id for r in group]
                for r in group:
                    failed_record_ids.add(r.record_id)

                all_violations.append(
                    RightsViolation(
                        rule_code="ERR_DUPLICATE_IMAGE_HASH",
                        severity=ValidationSeverity.BLOCKER,
                        message=f"Duplicate SHA-256 image hash '{h}' detected across {len(group)} records: {rec_ids}.",
                        record_id=group[0].record_id,
                        unresolved_requirement="Deduplicate dataset records sharing identical SHA-256 digest.",
                    )
                )
                unresolved_set.add("Deduplicate dataset records sharing identical SHA-256 digest.")

                # Check for conflicting provenance
                labels = {r.label for r in group}
                sources = {r.source_dataset for r in group}
                licenses = {r.original_license for r in group}
                creators = {r.source_creator for r in group}
                categories = {r.manipulation_category for r in group}

                has_conflict = len(labels) > 1 or len(sources) > 1 or len(licenses) > 1 or len(categories) > 1
                if has_conflict:
                    conflict_info = {
                        "sha256": h,
                        "record_ids": rec_ids,
                        "labels": [lbl.value for lbl in labels],
                        "source_datasets": list(sources),
                        "licenses": list(licenses),
                        "manipulation_categories": [cat.value for cat in categories],
                    }
                    conflicting_provenance.append(conflict_info)
                    all_violations.append(
                        RightsViolation(
                            rule_code="ERR_CONFLICTING_PROVENANCE",
                            severity=ValidationSeverity.BLOCKER,
                            message=(
                                f"Identical image hash '{h}' has contradictory provenance: "
                                f"labels={list(labels)}, sources={list(sources)}, licenses={list(licenses)}."
                            ),
                            record_id=group[0].record_id,
                            unresolved_requirement="Resolve contradictory provenance and purge erroneous record.",
                        )
                    )
                    unresolved_set.add("Resolve contradictory provenance and purge erroneous record.")

        total_count = len(records)
        failed_count = len(failed_record_ids)
        passed_count = total_count - failed_count
        has_blocking_violations = any(v.severity == ValidationSeverity.BLOCKER for v in all_violations)

        is_valid = not has_blocking_violations

        if is_valid and len(all_violations) == 0:
            status = RightsReviewStatus.APPROVED_FOR_TARGET
        elif is_valid and len(all_violations) > 0:
            status = RightsReviewStatus.ACTION_REQUIRED_UNRESOLVED
        else:
            status = RightsReviewStatus.REJECTED

        technical_summary = {
            "total_records": total_count,
            "passed_records": passed_count,
            "failed_records": failed_count,
            "target_usage": target_usage.value,
            "manifest_id": manifest_id,
            "dataset_name": dataset_name,
            "is_aggregated": is_aggregated,
            "general_license": general_license,
            "num_violations": len(all_violations),
            "num_blockers": sum(1 for v in all_violations if v.severity == ValidationSeverity.BLOCKER),
            "num_warnings": sum(1 for v in all_violations if v.severity == ValidationSeverity.WARNING),
            "duplicate_hash_count": len(duplicate_hashes),
            "conflicting_provenance_count": len(conflicting_provenance),
        }

        return ValidationReport(
            is_valid=is_valid,
            status=status,
            target_usage=target_usage,
            total_records=total_count,
            passed_records=passed_count,
            failed_records=failed_count,
            violations=all_violations,
            unresolved_approval_requirements=sorted(list(unresolved_set)),
            duplicate_hashes_found=duplicate_hashes,
            conflicting_provenance_records=conflicting_provenance,
            technical_summary=technical_summary,
        )
