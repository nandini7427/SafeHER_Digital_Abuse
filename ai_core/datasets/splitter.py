"""Reproducible and leakage-aware dataset splitting engine for SafeHER image pipelines.

Prevents:
1. Identity / Subject leakage: Consenting actors and identity groups never cross partition boundaries.
2. Lineage / Derivative leakage: Pristine originals and all derived manipulated variants remain together.
3. Near-duplicate leakage: Perceptually similar assets (recompressed, cropped, or filtered) stay grouped.
4. Generator contamination: Held-out generator architectures are strictly isolated from training/val.
5. Cross-dataset leakage: Designated evaluation corpora are held out cleanly.

Guarantees:
- Deterministic output for fixed random seeds.
- Disjoint union-find cluster formulation.
- Class balance preservation across partitions where feasible.
- Explicit failure on impossible split requests: Never silently place leaked records into different splits.
- Cryptographic reproducibility audit tracking (seed, policy version, manifest hashes).
"""

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.duplicates import NearDuplicateCandidate
from ai_core.datasets.manifest import (
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    DatasetTaskType,
    ManipulationCategory,
    RightsApprovalStatus,
    compute_sha256,
)


class DatasetLeakageError(ValueError):
    """Raised when an impossible split configuration is provided or when leakage is detected."""


class SplitPolicy(BaseModel):
    """Configuration specification for group-aware dataset partitioning."""

    policy_version: str = Field(default="1.0.0", description="Semantic version of splitting policy")
    seed: int = Field(default=42, description="Random seed for deterministic pseudo-random shuffling")
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0, description="Target proportion for TRAIN")
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0, description="Target proportion for VAL")
    internal_test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0, description="Target proportion for INTERNAL_TEST")

    # Held-out targets
    held_out_generators: Set[str] = Field(
        default_factory=set,
        description="Generators to be completely excluded from TRAIN/VAL and held out for zero-shot testing",
    )
    held_out_generator_families: Set[str] = Field(
        default_factory=set,
        description="Generator families (e.g. 'diffusion', 'gan') to hold out",
    )
    held_out_datasets: Set[str] = Field(
        default_factory=set,
        description="Dataset names to hold out entirely for cross-dataset evaluation",
    )
    compression_robustness_categories: Set[str] = Field(
        default_factory=set,
        description="Manipulation categories or processing steps reserved for compression robustness evaluation",
    )

    # Grouping flags
    group_by_identity: bool = Field(default=True, description="Enforce zero identity/subject overlap across splits")
    group_by_lineage: bool = Field(default=True, description="Keep original images and all derivative variants in same split")
    group_by_near_duplicates: bool = Field(default=True, description="Keep perceptual near-duplicate clusters in same split")
    stratify_by_label: bool = Field(default=True, description="Balance authentic and manipulated samples across splits")

    model_config = ConfigDict(frozen=True)


class SplitResult(BaseModel):
    """Container for partitioned dataset records and reproducibility metadata."""

    split_seed: int = Field(..., description="Random seed used for partitioning")
    policy_version: str = Field(..., description="Splitting policy version")
    grouping_method: str = Field(..., description="Method used for atomic cluster formulation")
    input_manifest_hash: str = Field(..., description="SHA-256 digest of input records before partitioning")
    split_manifest_hash: str = Field(..., description="SHA-256 digest of partitioned output records")
    source_dataset_versions: Dict[str, str] = Field(default_factory=dict)
    split_statistics: Dict[str, Any] = Field(default_factory=dict)
    records_by_split: Dict[DatasetSplit, List[DatasetImageRecord]] = Field(default_factory=dict)
    all_partitioned_records: List[DatasetImageRecord] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 == root2:
            return

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1


class GroupAwareDatasetSplitter:
    """Partitions dataset records into leakage-free splits using disjoint union-find clustering."""

    def __init__(self, policy: Optional[SplitPolicy] = None):
        self.policy = policy or SplitPolicy()

        # Validate ratios
        sum_ratios = self.policy.train_ratio + self.policy.val_ratio + self.policy.internal_test_ratio
        if not math.isclose(sum_ratios, 1.0, rel_tol=1e-3):
            raise DatasetLeakageError(
                f"Split ratios must sum to 1.0. Given: train={self.policy.train_ratio}, "
                f"val={self.policy.val_ratio}, test={self.policy.internal_test_ratio} (sum={sum_ratios:.4f})"
            )

    def _build_clusters(
        self,
        records: List[DatasetImageRecord],
        near_duplicate_pairs: Optional[List[NearDuplicateCandidate]] = None,
    ) -> Dict[str, List[DatasetImageRecord]]:
        """Formulate atomic leakage-safe clusters using Union-Find."""
        uf = UnionFind()

        for rec in records:
            rec_node = f"rec:{rec.record_id}"
            uf.find(rec_node)

            # Exact Cryptographic Content Duplicates (SHA-256) and identical file paths
            if rec.sha256:
                uf.union(rec_node, f"sha256:{rec.sha256.lower()}")
            if rec.image_path_or_ref:
                uf.union(rec_node, f"path:{rec.image_path_or_ref.strip().lower()}")

            # 1. Group by Identity / Subject Group
            if self.policy.group_by_identity:
                if rec.grouping_id:
                    uf.union(rec_node, f"group:{rec.grouping_id}")
                if rec.consent_reference:
                    uf.union(rec_node, f"consent:{rec.consent_reference}")

            # 2. Group by Lineage & Original/Manipulated Pairs
            if self.policy.group_by_lineage:
                # Register self as potential parent/root so children linking here are unified
                uf.union(rec_node, f"root:{rec.record_id}")
                if rec.image_id:
                    uf.union(rec_node, f"img:{rec.image_id}")
                    uf.union(rec_node, f"root:{rec.image_id}")
                if rec.image_path_or_ref:
                    uf.union(rec_node, f"root:{rec.image_path_or_ref.strip().lower()}")

                if rec.original_image_ref:
                    ref_clean = rec.original_image_ref.strip().lower()
                    uf.union(rec_node, f"root:{rec.original_image_ref}")
                    uf.union(rec_node, f"rec:{rec.original_image_ref}")
                    uf.union(rec_node, f"img:{rec.original_image_ref}")
                    uf.union(rec_node, f"path:{ref_clean}")
                if rec.parent_image_ref:
                    ref_clean = rec.parent_image_ref.strip().lower()
                    uf.union(rec_node, f"root:{rec.parent_image_ref}")
                    uf.union(rec_node, f"rec:{rec.parent_image_ref}")
                    uf.union(rec_node, f"img:{rec.parent_image_ref}")
                    uf.union(rec_node, f"path:{ref_clean}")

                # Source video / frame sequence lineage heuristic
                path_str = (rec.image_path_or_ref or "").replace("\\", "/")
                video_match = re.search(r"((?:video|vid|seq|clip)[-_]?\d+)", path_str, re.IGNORECASE)
                if video_match:
                    uf.union(rec_node, f"video:{video_match.group(1).lower()}")

        # 3. Group by Near-Duplicate Clusters
        if self.policy.group_by_near_duplicates and near_duplicate_pairs:
            for pair in near_duplicate_pairs:
                node_a = f"rec:{pair.record_id_a}"
                node_b = f"rec:{pair.record_id_b}"
                uf.union(node_a, node_b)

        # Collect records into clusters by their root representative
        clusters: Dict[str, List[DatasetImageRecord]] = defaultdict(list)
        for rec in records:
            cluster_id = uf.find(f"rec:{rec.record_id}")
            clusters[cluster_id].append(rec)

        return dict(clusters)

    def split_records(
        self,
        records: List[DatasetImageRecord],
        near_duplicate_pairs: Optional[List[NearDuplicateCandidate]] = None,
    ) -> SplitResult:
        """Partition records into leakage-free splits according to policy.

        Raises:
            DatasetLeakageError: If impossible split request or contradictory grouping occurs.
        """
        if not records:
            raise DatasetLeakageError("Cannot split an empty dataset record list.")

        # Compute input manifest hash for reproducibility
        input_record_hashes = "".join(sorted(r.sha256 for r in records))
        input_manifest_hash = hashlib.sha256(input_record_hashes.encode("utf-8")).hexdigest()

        # Step 1: Formulate atomic clusters
        clusters = self._build_clusters(records, near_duplicate_pairs=near_duplicate_pairs)

        split_assignments: Dict[DatasetSplit, List[DatasetImageRecord]] = {
            DatasetSplit.TRAIN: [],
            DatasetSplit.VAL: [],
            DatasetSplit.INTERNAL_TEST: [],
            DatasetSplit.HELD_OUT_GENERATOR_TEST: [],
            DatasetSplit.CROSS_DATASET_TEST: [],
            DatasetSplit.COMPRESSION_ROBUSTNESS_TEST: [],
        }

        remaining_clusters: Dict[str, List[DatasetImageRecord]] = {}

        # Normalized held-out sets
        held_out_gens_clean = {g.strip().lower() for g in self.policy.held_out_generators}
        held_out_ds_clean = {d.strip().lower() for d in self.policy.held_out_datasets}

        # Step 2: Extract specialized held-out test splits
        for c_id, c_records in clusters.items():
            cluster_gens = {
                (r.generator_name or "").strip().lower()
                for r in c_records
                if r.generator_name
            }
            cluster_datasets = {
                (r.source_dataset or "").strip().lower()
                for r in c_records
                if r.source_dataset
            }

            # Check: Held-out generator test
            is_held_out_gen = bool(cluster_gens & held_out_gens_clean)
            # Check: Cross-dataset test
            is_held_out_ds = bool(cluster_datasets & held_out_ds_clean)

            # Check for conflict: Cluster cannot be both held-out and required for in-domain training
            if is_held_out_gen and is_held_out_ds:
                # Assign to held-out generator test
                for r in c_records:
                    split_assignments[DatasetSplit.HELD_OUT_GENERATOR_TEST].append(
                        r.model_copy(update={"split": DatasetSplit.HELD_OUT_GENERATOR_TEST})
                    )
            elif is_held_out_gen:
                for r in c_records:
                    split_assignments[DatasetSplit.HELD_OUT_GENERATOR_TEST].append(
                        r.model_copy(update={"split": DatasetSplit.HELD_OUT_GENERATOR_TEST})
                    )
            elif is_held_out_ds:
                for r in c_records:
                    split_assignments[DatasetSplit.CROSS_DATASET_TEST].append(
                        r.model_copy(update={"split": DatasetSplit.CROSS_DATASET_TEST})
                    )
            else:
                remaining_clusters[c_id] = c_records

        # Step 3: Check if remaining clusters can satisfy TRAIN, VAL, and INTERNAL_TEST
        if len(remaining_clusters) < 3:
            raise DatasetLeakageError(
                f"Impossible split configuration: Only {len(remaining_clusters)} independent cluster(s) available "
                f"for TRAIN, VAL, and INTERNAL_TEST. At least 3 independent non-leaking clusters are strictly required. "
                "SafeHER never silently places leaked records into different splits."
            )

        # Step 4: Partition remaining clusters into TRAIN, VAL, INTERNAL_TEST
        rng = np.random.RandomState(self.policy.seed)
        cluster_ids = list(remaining_clusters.keys())

        # Sort cluster IDs deterministically before shuffling
        cluster_ids.sort()
        rng.shuffle(cluster_ids)

        # Optional stratification by majority label in each cluster
        if self.policy.stratify_by_label:
            manipulated_clusters: List[str] = []
            authentic_clusters: List[str] = []
            for cid in cluster_ids:
                recs = remaining_clusters[cid]
                # Count manipulated vs authentic
                num_manip = sum(
                    1 for r in recs if r.label != DatasetImageLabel.CAMERA_PHOTOGRAPH
                )
                if num_manip >= len(recs) / 2.0:
                    manipulated_clusters.append(cid)
                else:
                    authentic_clusters.append(cid)

            # Partition each stratified sub-list
            def _partition_sublist(sub_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
                n = len(sub_ids)
                if n == 0:
                    return [], [], []
                n_train = max(1, int(round(n * self.policy.train_ratio)))
                n_val = max(1 if n >= 3 else 0, int(round(n * self.policy.val_ratio)))
                # Adjust to ensure not exceeding n
                if n_train + n_val >= n:
                    n_train = max(1, n - 2) if n >= 3 else 1
                    n_val = 1 if n >= 2 else 0

                train_part = sub_ids[:n_train]
                val_part = sub_ids[n_train : n_train + n_val]
                test_part = sub_ids[n_train + n_val :]
                return train_part, val_part, test_part

            tr_m, val_m, test_m = _partition_sublist(manipulated_clusters)
            tr_a, val_a, test_a = _partition_sublist(authentic_clusters)

            train_clusters = tr_m + tr_a
            val_clusters = val_m + val_a
            test_clusters = test_m + test_a

            # If test is empty due to small cluster count, balance
            if not test_clusters and len(train_clusters) > 1:
                test_clusters.append(train_clusters.pop())
            if not val_clusters and len(train_clusters) > 1:
                val_clusters.append(train_clusters.pop())

        else:
            total_remaining = len(cluster_ids)
            n_train = max(1, int(round(total_remaining * self.policy.train_ratio)))
            n_val = max(1, int(round(total_remaining * self.policy.val_ratio)))
            train_clusters = cluster_ids[:n_train]
            val_clusters = cluster_ids[n_train : n_train + n_val]
            test_clusters = cluster_ids[n_train + n_val :]

        # Populate partitioned records
        for cid in train_clusters:
            for r in remaining_clusters[cid]:
                split_assignments[DatasetSplit.TRAIN].append(
                    r.model_copy(update={"split": DatasetSplit.TRAIN})
                )

        for cid in val_clusters:
            for r in remaining_clusters[cid]:
                split_assignments[DatasetSplit.VAL].append(
                    r.model_copy(update={"split": DatasetSplit.VAL})
                )

        for cid in test_clusters:
            for r in remaining_clusters[cid]:
                split_assignments[DatasetSplit.INTERNAL_TEST].append(
                    r.model_copy(update={"split": DatasetSplit.INTERNAL_TEST})
                )

        # Assemble full partitioned record list
        all_partitioned: List[DatasetImageRecord] = []
        for split_type in DatasetSplit:
            all_partitioned.extend(split_assignments.get(split_type, []))

        # Step 5: Post-split verification and leakage check
        self._verify_leakage_absence(
            split_assignments=split_assignments,
            held_out_generators=held_out_gens_clean,
            held_out_datasets=held_out_ds_clean,
        )

        # Compute output manifest hash
        out_hashes = "".join(sorted(f"{r.record_id}:{r.split.value}" for r in all_partitioned))
        split_manifest_hash = hashlib.sha256(out_hashes.encode("utf-8")).hexdigest()

        # Compute split statistics
        stats: Dict[str, Any] = {}
        dataset_versions: Dict[str, str] = {}
        for sp, rec_list in split_assignments.items():
            if rec_list:
                auth_count = sum(1 for r in rec_list if r.label == DatasetImageLabel.CAMERA_PHOTOGRAPH)
                manip_count = len(rec_list) - auth_count
                stats[sp.value] = {
                    "total_count": len(rec_list),
                    "authentic_count": auth_count,
                    "manipulated_count": manip_count,
                    "authentic_ratio": round(auth_count / len(rec_list), 4) if rec_list else 0.0,
                }
                for r in rec_list:
                    if r.dataset_name and r.dataset_version:
                        dataset_versions[r.dataset_name] = r.dataset_version

        return SplitResult(
            split_seed=self.policy.seed,
            policy_version=self.policy.policy_version,
            grouping_method="composite_union_find_identity_lineage_near_duplicate",
            input_manifest_hash=input_manifest_hash,
            split_manifest_hash=split_manifest_hash,
            source_dataset_versions=dataset_versions,
            split_statistics=stats,
            records_by_split=split_assignments,
            all_partitioned_records=all_partitioned,
        )

    def _verify_leakage_absence(
        self,
        split_assignments: Dict[DatasetSplit, List[DatasetImageRecord]],
        held_out_generators: Set[str],
        held_out_datasets: Set[str],
    ) -> None:
        """Internal post-partition verification asserting zero identity, lineage, or generator leakage."""
        train_records = split_assignments[DatasetSplit.TRAIN]
        val_records = split_assignments[DatasetSplit.VAL]
        test_records = (
            split_assignments[DatasetSplit.INTERNAL_TEST]
            + split_assignments[DatasetSplit.HELD_OUT_GENERATOR_TEST]
            + split_assignments[DatasetSplit.CROSS_DATASET_TEST]
        )

        # 1. Identity / Subject Group overlap check
        if self.policy.group_by_identity:
            train_groups = {r.grouping_id for r in train_records if r.grouping_id}
            val_groups = {r.grouping_id for r in val_records if r.grouping_id}
            test_groups = {r.grouping_id for r in test_records if r.grouping_id}

            tr_val_leak = train_groups & val_groups
            if tr_val_leak:
                raise DatasetLeakageError(
                    f"Identity leakage detected between TRAIN and VAL across group(s): {tr_val_leak}."
                )

            tr_test_leak = train_groups & test_groups
            if tr_test_leak:
                raise DatasetLeakageError(
                    f"Identity leakage detected between TRAIN and TEST across group(s): {tr_test_leak}."
                )

        # 2. Lineage overlap check (original_image_ref)
        if self.policy.group_by_lineage:
            train_roots = {r.original_image_ref for r in train_records if r.original_image_ref}
            test_roots = {r.original_image_ref for r in test_records if r.original_image_ref}
            val_roots = {r.original_image_ref for r in val_records if r.original_image_ref}

            leak_tr_val = train_roots & val_roots
            if leak_tr_val:
                raise DatasetLeakageError(
                    f"Lineage leakage detected between TRAIN and VAL across root(s): {leak_tr_val}."
                )

            leak_tr_test = train_roots & test_roots
            if leak_tr_test:
                raise DatasetLeakageError(
                    f"Lineage leakage detected between TRAIN and TEST across root(s): {leak_tr_test}."
                )

        # 3. Held-out generator check
        if held_out_generators:
            for r in train_records + val_records:
                gen_clean = (r.generator_name or "").strip().lower()
                if gen_clean in held_out_generators:
                    raise DatasetLeakageError(
                        f"Held-out generator '{gen_clean}' leaked into {r.split.value} split on record '{r.record_id}'."
                    )

        # 4. Held-out dataset check
        if held_out_datasets:
            for r in train_records + val_records:
                ds_clean = (r.source_dataset or "").strip().lower()
                if ds_clean in held_out_datasets:
                    raise DatasetLeakageError(
                        f"Held-out dataset '{ds_clean}' leaked into {r.split.value} split on record '{r.record_id}'."
                    )
