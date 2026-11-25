"""Leakage detection and dataset split auditing engine for SafeHER.

Verifies:
1. Identity / Subject Group separation: Consenting actors and identity tokens never cross splits.
2. Lineage / Derivative integrity: Parent images, roots, and manipulated derivatives remain co-located.
3. Near-duplicate isolation: Perceptual hash near-duplicate candidate pairs never span train vs test/val.
4. Generator hold-out integrity: Held-out generator outputs are 100% quarantined to test splits.
5. Cross-dataset hold-out integrity: Entire evaluation datasets remain strictly unreferenced by training.
6. Test accessibility isolation: Independent test sets remain inaccessible to training and threshold tuning.
"""

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.duplicates import NearDuplicateCandidate
from ai_core.datasets.manifest import (
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    TargetUsageScope,
)


class LeakageType(str, Enum):
    """Categorical taxonomy of dataset split leakage."""

    EXACT_DUPLICATE_LEAKAGE = "EXACT_DUPLICATE_LEAKAGE"
    NEAR_DUPLICATE_LEAKAGE = "NEAR_DUPLICATE_LEAKAGE"
    IDENTITY_LEAKAGE = "IDENTITY_LEAKAGE"
    LINEAGE_DERIVATIVE_LEAKAGE = "LINEAGE_DERIVATIVE_LEAKAGE"
    SOURCE_VIDEO_LEAKAGE = "SOURCE_VIDEO_LEAKAGE"
    HELD_OUT_GENERATOR_LEAKAGE = "HELD_OUT_GENERATOR_LEAKAGE"
    CROSS_DATASET_LEAKAGE = "CROSS_DATASET_LEAKAGE"
    METADATA_LABEL_LEAKAGE = "METADATA_LABEL_LEAKAGE"
    WATERMARK_LABEL_LEAKAGE = "WATERMARK_LABEL_LEAKAGE"
    DATASET_SOURCE_CONFOUNDING = "DATASET_SOURCE_CONFOUNDING"
    AUGMENTATION_LEAKAGE = "AUGMENTATION_LEAKAGE"
    TEST_ACCESSIBILITY_VIOLATION = "TEST_ACCESSIBILITY_VIOLATION"
    VALIDATION_TEST_CONTAMINATION = "VALIDATION_TEST_CONTAMINATION"


class LeakageViolation(BaseModel):
    """Structured report of a leakage finding or split policy breach."""

    leakage_type: LeakageType = Field(..., description="Category of leakage detected")
    severity: str = Field(default="BLOCKER", description="'BLOCKER' or 'WARNING'")
    record_ids: List[str] = Field(..., description="IDs of records implicated in the leakage")
    splits_involved: List[str] = Field(..., description="Dataset splits across which leakage occurred")
    details: str = Field(..., description="Human-readable description of how leakage occurred")
    remediation: str = Field(..., description="Actionable fix to re-partition or eliminate leakage")

    model_config = ConfigDict(frozen=True)


class LeakageAuditReport(BaseModel):
    """Comprehensive leakage audit report detailing split integrity and cross-partition contamination."""

    audit_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time of audit execution",
    )
    is_leakage_free: bool = Field(..., description="True if zero blocking leakage violations exist")
    total_records_audited: int = Field(..., ge=0)
    total_violations_found: int = Field(default=0)
    violations: List[LeakageViolation] = Field(default_factory=list)
    split_summary: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def to_markdown(self) -> str:
        """Render a GitHub-style Markdown leakage audit report."""
        status_text = "LEAKAGE-FREE (PASSED)" if self.is_leakage_free else "LEAKAGE DETECTED (FAILED)"
        lines = [
            "# SafeHER Dataset Leakage Audit Report",
            "",
            f"**Audit Result:** {status_text}  ",
            f"**Audit Timestamp:** {self.audit_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Total Records Evaluated:** {self.total_records_audited}  ",
            f"**Leakage Violations Detected:** {self.total_violations_found}  ",
            "",
            "## Split Breakdown and Class Balance",
            "",
            "| Split | Total Records | Authentic Count | Manipulated Count | Authentic Ratio |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for split_name, stats in self.split_summary.items():
            tot = stats.get("total_count", 0)
            auth = stats.get("authentic_count", 0)
            manip = stats.get("manipulated_count", 0)
            ratio = stats.get("authentic_ratio", 0.0)
            lines.append(f"| `{split_name}` | {tot} | {auth} | {manip} | {ratio:.2%} |")

        lines.append("")

        if self.violations:
            lines.append("## Leakage Violations Detail")
            lines.append("")
            lines.append("| Severity | Leakage Type | Splits Involved | Implicated Records | Details |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for v in self.violations:
                rec_str = ", ".join(v.record_ids[:4]) + ("..." if len(v.record_ids) > 4 else "")
                splits_str = ", ".join(v.splits_involved)
                lines.append(f"| `{v.severity}` | `{v.leakage_type.value}` | `{splits_str}` | {rec_str} | {v.details} |")
            lines.append("")

        return "\n".join(lines)


class LeakageDetector:
    """Audits dataset record partitioning for cross-split contamination and data leakage."""

    def __init__(
        self,
        held_out_generators: Optional[Set[str]] = None,
        held_out_datasets: Optional[Set[str]] = None,
    ):
        self.held_out_generators = {g.strip().lower() for g in (held_out_generators or set())}
        self.held_out_datasets = {d.strip().lower() for d in (held_out_datasets or set())}

    def audit_splits(
        self,
        records: List[DatasetImageRecord],
        near_duplicate_pairs: Optional[List[NearDuplicateCandidate]] = None,
    ) -> LeakageAuditReport:
        """Run thorough audit across all split partitions for identity, lineage, near-duplicate, and domain leakage."""
        violations: List[LeakageViolation] = []

        # Group records by assigned split
        split_to_records: Dict[DatasetSplit, List[DatasetImageRecord]] = defaultdict(list)
        for r in records:
            split_to_records[r.split].append(r)

        train_records = split_to_records.get(DatasetSplit.TRAIN, [])
        val_records = split_to_records.get(DatasetSplit.VAL, [])
        test_records = (
            split_to_records.get(DatasetSplit.TEST, [])
            + split_to_records.get(DatasetSplit.INTERNAL_TEST, [])
            + split_to_records.get(DatasetSplit.HELD_OUT_GENERATOR_TEST, [])
            + split_to_records.get(DatasetSplit.CROSS_DATASET_TEST, [])
            + split_to_records.get(DatasetSplit.COMPRESSION_ROBUSTNESS_TEST, [])
        )

        train_record_ids = {r.record_id for r in train_records}
        val_record_ids = {r.record_id for r in val_records}
        test_record_ids = {r.record_id for r in test_records}

        # ----------------------------------------------------------------------
        # Check 1: Identity / Subject Group & Consent Token Leakage
        # ----------------------------------------------------------------------
        group_to_splits: Dict[str, Dict[DatasetSplit, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            if r.grouping_id:
                group_to_splits[f"group:{r.grouping_id}"][r.split].append(r.record_id)
            if r.consent_reference:
                group_to_splits[f"consent:{r.consent_reference}"][r.split].append(r.record_id)

        for gid, splits_map in group_to_splits.items():
            active_splits = [sp for sp, r_list in splits_map.items() if r_list]
            has_train = DatasetSplit.TRAIN in active_splits
            has_eval = any(sp != DatasetSplit.TRAIN for sp in active_splits)

            if has_train and has_eval:
                all_involved: List[str] = []
                for r_list in splits_map.values():
                    all_involved.extend(r_list)

                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.IDENTITY_LEAKAGE,
                        severity="BLOCKER",
                        record_ids=all_involved,
                        splits_involved=[sp.value for sp in active_splits],
                        details=(
                            f"Identity / subject group or consent token '{gid}' appears in multiple conflicting splits "
                            f"({[sp.value for sp in active_splits]}). Memorization of actor likeness will taint evaluation."
                        ),
                        remediation="Re-partition using group-aware clustering to ensure entire identity group is in one split.",
                    )
                )

        # ----------------------------------------------------------------------
        # Check 1b: Exact Cryptographic & Path Content Duplicates Across Splits
        # ----------------------------------------------------------------------
        sha256_to_splits: Dict[str, Dict[DatasetSplit, List[str]]] = defaultdict(lambda: defaultdict(list))
        path_to_splits: Dict[str, Dict[DatasetSplit, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            if r.sha256:
                sha256_to_splits[r.sha256.lower()][r.split].append(r.record_id)
            if r.image_path_or_ref:
                path_to_splits[r.image_path_or_ref.strip().lower()][r.split].append(r.record_id)

        for digest, splits_map in sha256_to_splits.items():
            active_splits = [sp for sp, r_list in splits_map.items() if r_list]
            has_train = DatasetSplit.TRAIN in active_splits
            has_eval = any(sp != DatasetSplit.TRAIN for sp in active_splits)
            if has_train and has_eval:
                all_involved = []
                for r_list in splits_map.values():
                    all_involved.extend(r_list)
                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.EXACT_DUPLICATE_LEAKAGE,
                        severity="BLOCKER",
                        record_ids=all_involved,
                        splits_involved=[sp.value for sp in active_splits],
                        details=(
                            f"Exact cryptographic duplicate image (SHA-256={digest[:16]}...) appears in "
                            f"multiple conflicting splits ({[sp.value for sp in active_splits]}). "
                            "Direct memorization of training pixels leaks into evaluation."
                        ),
                        remediation="Deduplicate exact duplicates or cluster identical SHA-256 digests into the same partition.",
                    )
                )

        for path_key, splits_map in path_to_splits.items():
            active_splits = [sp for sp, r_list in splits_map.items() if r_list]
            has_train = DatasetSplit.TRAIN in active_splits
            has_eval = any(sp != DatasetSplit.TRAIN for sp in active_splits)
            if has_train and has_eval:
                all_involved = []
                for r_list in splits_map.values():
                    all_involved.extend(r_list)
                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.EXACT_DUPLICATE_LEAKAGE,
                        severity="BLOCKER",
                        record_ids=all_involved,
                        splits_involved=[sp.value for sp in active_splits],
                        details=(
                            f"Identical file path or asset reference '{path_key}' appears in multiple conflicting splits "
                            f"({[sp.value for sp in active_splits]})."
                        ),
                        remediation="Ensure unique file references per split.",
                    )
                )

        # ----------------------------------------------------------------------
        # Check 2: Bidirectional Lineage / Derivative & Original-Manipulated Pairs
        # ----------------------------------------------------------------------
        # Forward registry of parent IDs/paths to splits
        id_to_split: Dict[str, DatasetSplit] = {}
        for r in records:
            id_to_split[r.record_id] = r.split
            if r.image_id:
                id_to_split[r.image_id] = r.split
            if r.image_path_or_ref:
                id_to_split[r.image_path_or_ref.strip().lower()] = r.split

        # 2a. Direct parent-child linkage check
        for r in records:
            for parent_ref in (r.original_image_ref, r.parent_image_ref):
                if not parent_ref:
                    continue
                clean_ref = parent_ref.strip()
                target_split = id_to_split.get(clean_ref) or id_to_split.get(clean_ref.lower())
                if target_split and target_split != r.split:
                    is_train_child = (r.split == DatasetSplit.TRAIN)
                    is_train_parent = (target_split == DatasetSplit.TRAIN)
                    if is_train_child != is_train_parent:
                        violations.append(
                            LeakageViolation(
                                leakage_type=LeakageType.LINEAGE_DERIVATIVE_LEAKAGE,
                                severity="BLOCKER",
                                record_ids=[r.record_id, clean_ref],
                                splits_involved=[r.split.value, target_split.value],
                                details=(
                                    f"Original/parent asset '{clean_ref}' in split '{target_split.value}' has "
                                    f"derivative child variant '{r.record_id}' in conflicting split '{r.split.value}'. "
                                    "Training on pristine parent background will cause memorization of test deepfakes."
                                ),
                                remediation="Keep all parent images and derived variants strictly in the same partition.",
                            )
                        )

        # 2b. Common root image references across splits
        root_to_splits: Dict[str, Dict[DatasetSplit, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            for root_ref in (r.original_image_ref, r.parent_image_ref):
                if root_ref:
                    root_to_splits[root_ref.strip().lower()][r.split].append(r.record_id)

        for root_ref, splits_map in root_to_splits.items():
            active_splits = [sp for sp, r_list in splits_map.items() if r_list]
            has_train = DatasetSplit.TRAIN in active_splits
            has_eval = any(sp != DatasetSplit.TRAIN for sp in active_splits)

            if has_train and has_eval:
                all_involved = []
                for r_list in splits_map.values():
                    all_involved.extend(r_list)

                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.LINEAGE_DERIVATIVE_LEAKAGE,
                        severity="BLOCKER",
                        record_ids=all_involved,
                        splits_involved=[sp.value for sp in active_splits],
                        details=(
                            f"Original ancestor / root image '{root_ref}' has derivative variants spanning "
                            f"multiple splits ({[sp.value for sp in active_splits]}). The model will recognize pristine training backgrounds."
                        ),
                        remediation="Keep all parent images and derived variants strictly in the same partition.",
                    )
                )

        # ----------------------------------------------------------------------
        # Check 2c: Source Video Sequence Frame Leakage
        # ----------------------------------------------------------------------
        video_to_splits: Dict[str, Dict[DatasetSplit, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            path_str = (r.image_path_or_ref or "").replace("\\", "/")
            video_match = re.search(r"((?:video|vid|seq|clip)[-_]?\d+)", path_str, re.IGNORECASE)
            if video_match:
                vid_name = video_match.group(1).lower()
                video_to_splits[vid_name][r.split].append(r.record_id)

        for vid_name, splits_map in video_to_splits.items():
            active_splits = [sp for sp, r_list in splits_map.items() if r_list]
            has_train = DatasetSplit.TRAIN in active_splits
            has_eval = any(sp != DatasetSplit.TRAIN for sp in active_splits)
            if has_train and has_eval:
                all_involved = []
                for r_list in splits_map.values():
                    all_involved.extend(r_list)
                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.SOURCE_VIDEO_LEAKAGE,
                        severity="BLOCKER",
                        record_ids=all_involved,
                        splits_involved=[sp.value for sp in active_splits],
                        details=(
                            f"Video sequence '{vid_name}' has video frames distributed across multiple conflicting splits "
                            f"({[sp.value for sp in active_splits]}). Adjacent video frames share near-identical camera sensor noise."
                        ),
                        remediation="Group video frame sequences by video identifier using group_by_lineage.",
                    )
                )

        # ----------------------------------------------------------------------
        # Check 3: Near-Duplicate Leakage
        # ----------------------------------------------------------------------
        if near_duplicate_pairs:
            rec_to_split: Dict[str, DatasetSplit] = {r.record_id: r.split for r in records}
            for pair in near_duplicate_pairs:
                split_a = rec_to_split.get(pair.record_id_a)
                split_b = rec_to_split.get(pair.record_id_b)

                if split_a and split_b and split_a != split_b:
                    is_train_a = split_a == DatasetSplit.TRAIN
                    is_train_b = split_b == DatasetSplit.TRAIN
                    # Leakage if one is train and the other is not
                    if is_train_a != is_train_b:
                        violations.append(
                            LeakageViolation(
                                leakage_type=LeakageType.NEAR_DUPLICATE_LEAKAGE,
                                severity="BLOCKER",
                                record_ids=[pair.record_id_a, pair.record_id_b],
                                splits_involved=[split_a.value, split_b.value],
                                details=(
                                    f"Near-duplicate candidate pair ({pair.record_id_a} vs {pair.record_id_b}, "
                                    f"Hamming distance={pair.hamming_distance}/64) crosses split boundary "
                                    f"between '{split_a.value}' and '{split_b.value}'."
                                ),
                                remediation="Union near-duplicate candidate clusters into the same partition before splitting.",
                            )
                        )

        # ----------------------------------------------------------------------
        # Check 4: Held-Out Generator Leakage
        # ----------------------------------------------------------------------
        if self.held_out_generators:
            for r in train_records + val_records:
                gen_clean = (r.generator_name or "").strip().lower()
                if gen_clean in self.held_out_generators:
                    violations.append(
                        LeakageViolation(
                            leakage_type=LeakageType.HELD_OUT_GENERATOR_LEAKAGE,
                            severity="BLOCKER",
                            record_ids=[r.record_id],
                            splits_involved=[r.split.value],
                            details=(
                                f"Record '{r.record_id}' generated by held-out generator '{gen_clean}' "
                                f"found in training/val partition '{r.split.value}'. Invalidates zero-shot generalization testing."
                            ),
                            remediation=f"Quarantine all '{gen_clean}' assets exclusively to HELD_OUT_GENERATOR_TEST.",
                        )
                    )

        # ----------------------------------------------------------------------
        # Check 5: Cross-Dataset Leakage
        # ----------------------------------------------------------------------
        if self.held_out_datasets:
            for r in train_records + val_records:
                ds_clean = (r.source_dataset or "").strip().lower()
                if ds_clean in self.held_out_datasets:
                    violations.append(
                        LeakageViolation(
                            leakage_type=LeakageType.CROSS_DATASET_LEAKAGE,
                            severity="BLOCKER",
                            record_ids=[r.record_id],
                            splits_involved=[r.split.value],
                            details=(
                                f"Record '{r.record_id}' from held-out dataset '{ds_clean}' "
                                f"found in training/val partition '{r.split.value}'. Compromises independent evaluation integrity."
                            ),
                            remediation=f"Quarantine all '{ds_clean}' records exclusively to CROSS_DATASET_TEST.",
                        )
                    )

        # ----------------------------------------------------------------------
        # Check 6: Dataset-Source Confounding
        # ----------------------------------------------------------------------
        # Check if source dataset in train or validation is completely confounded with the target label
        ds_label_counts: Dict[str, Dict[DatasetImageLabel, int]] = defaultdict(lambda: defaultdict(int))
        for r in train_records:
            ds_name = r.source_dataset or "unknown"
            ds_label_counts[ds_name][r.label] += 1

        for ds_name, l_counts in ds_label_counts.items():
            total_ds_samples = sum(l_counts.values())
            if total_ds_samples >= 10:
                # If a dataset in training has only 1 class and zero of any other class
                num_classes_represented = sum(1 for c in l_counts.values() if c > 0)
                if num_classes_represented == 1 and len(ds_label_counts) > 1:
                    only_label = next(k.value for k, v in l_counts.items() if v > 0)
                    violations.append(
                        LeakageViolation(
                            leakage_type=LeakageType.DATASET_SOURCE_CONFOUNDING,
                            severity="WARNING",
                            record_ids=[r.record_id for r in train_records if r.source_dataset == ds_name][:10],
                            splits_involved=[DatasetSplit.TRAIN.value],
                            details=(
                                f"Training source dataset '{ds_name}' has 100% {only_label} samples ({total_ds_samples} total). "
                                "The model may learn dataset-specific compression or sensor signatures instead of forensic anomalies."
                            ),
                            remediation="Balance each source dataset with both authentic and manipulated samples.",
                        )
                    )

        # ----------------------------------------------------------------------
        # Check 7: File-Path and Metadata Label Leakage
        # ----------------------------------------------------------------------
        leakage_patterns = [
            (re.compile(r"/(?:fake|real|manipulated|synthetic|deepfake|pristine|control)/", re.IGNORECASE), "Directory label segment"),
            (re.compile(r"[-_](?:fake|real|synth|manip|deepfake)[-_.]", re.IGNORECASE), "Filename label token"),
        ]
        leaking_path_records = []
        for r in records:
            path_str = f"/{r.image_path_or_ref.replace('\\', '/')}"
            for pat, desc in leakage_patterns:
                if pat.search(path_str):
                    leaking_path_records.append((r.record_id, r.image_path_or_ref, desc))
                    break

        if leaking_path_records:
            # We record a warning to ensure paths are never ingested as features
            violations.append(
                LeakageViolation(
                    leakage_type=LeakageType.METADATA_LABEL_LEAKAGE,
                    severity="WARNING",
                    record_ids=[rec_id for rec_id, _, _ in leaking_path_records[:10]],
                    splits_involved=list({r.split.value for r in records if r.record_id in {x[0] for x in leaking_path_records}}),
                    details=(
                        f"Found {len(leaking_path_records)} image references containing explicit label tokens in file paths "
                        f"(e.g. '{leaking_path_records[0][1]}'). Features must never read or extract embeddings from file paths."
                    ),
                    remediation="Ensure image dataset loaders only pass raw pixel tensors to models, never filesystem paths or EXIF.",
                )
            )

        # ----------------------------------------------------------------------
        # Check 8: Test Accessibility & Tuning Isolation
        # ----------------------------------------------------------------------
        for r in test_records:
            if r.record_id in train_record_ids:
                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.TEST_ACCESSIBILITY_VIOLATION,
                        severity="BLOCKER",
                        record_ids=[r.record_id],
                        splits_involved=[r.split.value, DatasetSplit.TRAIN.value],
                        details=(
                            f"Record '{r.record_id}' in test split '{r.split.value}' is also present in training split. "
                            "Test data must never be accessible during training."
                        ),
                        remediation="Quarantine test records strictly to evaluation splits.",
                    )
                )
            elif getattr(r, "training_eligibility", False) and r.split != DatasetSplit.TRAIN:
                violations.append(
                    LeakageViolation(
                        leakage_type=LeakageType.TEST_ACCESSIBILITY_VIOLATION,
                        severity="WARNING",
                        record_ids=[r.record_id],
                        splits_involved=[r.split.value],
                        details=(
                            f"Record '{r.record_id}' in test split '{r.split.value}' has training_eligibility=True. "
                            "In production manifests, test assets should have training_eligibility=False."
                        ),
                        remediation="Set training_eligibility=False on all test split assets.",
                    )
                )

        # Split statistics compilation
        split_summary: Dict[str, Dict[str, Any]] = {}
        for sp, rec_list in split_to_records.items():
            if rec_list:
                auth = sum(1 for r in rec_list if r.label == DatasetImageLabel.CAMERA_PHOTOGRAPH)
                manip = len(rec_list) - auth
                split_summary[sp.value] = {
                    "total_count": len(rec_list),
                    "authentic_count": auth,
                    "manipulated_count": manip,
                    "authentic_ratio": round(auth / len(rec_list), 4) if rec_list else 0.0,
                }

        is_clean = not any(v.severity == "BLOCKER" for v in violations)

        return LeakageAuditReport(
            is_leakage_free=is_clean,
            total_records_audited=len(records),
            total_violations_found=len(violations),
            violations=violations,
            split_summary=split_summary,
            metadata={
                "held_out_generators": list(self.held_out_generators),
                "held_out_datasets": list(self.held_out_datasets),
                "audited_split_count": len(split_to_records),
            },
        )
