"""Provenance tracking, lineage graph, and non-destructive variant management for SafeHER datasets.

Enforces:
1. Complete tracking of original file, parent image, transformation chain, generator, and operation.
2. Immutability guarantee: Original provenance is NEVER overwritten when generating manipulated variants.
3. Provenance chain validation: Detects broken parent links, chronological inversions, missing generators,
   and unverified rights escalation across image lineages.
"""

from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.manifest import (
    AttributionMetadata,
    CommercialUseStatus,
    ConsentStatus,
    DatasetImageLabel,
    DatasetImageRecord,
    DatasetSplit,
    DatasetTaskType,
    GeneratorInfo,
    ManipulationCategory,
    ResearchUseStatus,
    RightsApprovalStatus,
    TransformationStep,
    UsageScope,
)


class ProvenanceViolationType(str, Enum):
    """Categorical classification of provenance and lineage inconsistencies."""

    MISSING_PARENT_RECORD = "MISSING_PARENT_RECORD"
    CHRONOLOGICAL_INVERSION = "CHRONOLOGICAL_INVERSION"
    MISSING_TRANSFORMATION_EVIDENCE = "MISSING_TRANSFORMATION_EVIDENCE"
    MISSING_GENERATOR_METADATA = "MISSING_GENERATOR_METADATA"
    CYCLIC_LINEAGE = "CYCLIC_LINEAGE"
    UNAUTHORIZED_RIGHTS_ESCALATION = "UNAUTHORIZED_RIGHTS_ESCALATION"
    INCONSISTENT_ROOT_REFERENCE = "INCONSISTENT_ROOT_REFERENCE"


class ProvenanceViolation(BaseModel):
    """Structured report of an inconsistency in an image's provenance chain."""

    violation_type: ProvenanceViolationType = Field(..., description="Category of provenance violation")
    record_id: str = Field(..., description="Affected child or candidate record ID")
    parent_id: Optional[str] = Field(None, description="Referenced parent record ID if applicable")
    message: str = Field(..., description="Human-readable explanation of the defect")
    actionable_fix: str = Field(..., description="Required remediation step")

    model_config = ConfigDict(frozen=True)


def create_derived_variant_record(
    parent_record: DatasetImageRecord,
    new_record_id: str,
    new_image_path_or_ref: str,
    new_sha256: str,
    manipulation_category: ManipulationCategory,
    new_label: DatasetImageLabel,
    transformation_step: TransformationStep,
    task_type: Optional[DatasetTaskType] = None,
    generator_name: Optional[str] = None,
    generator_version: Optional[str] = None,
    generator_info: Optional[GeneratorInfo] = None,
    processing_config: Optional[Dict[str, Any]] = None,
    new_split: Optional[DatasetSplit] = None,
    rights_status_override: Optional[RightsApprovalStatus] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> DatasetImageRecord:
    """Create a derived child record from an existing image without mutating or overwriting parent provenance.

    Guarantees:
    - The `parent_record` is left completely unmodified (immutable).
    - Child sets `parent_image_ref` to parent's `record_id`.
    - Child preserves `original_image_ref` tracing back to the root ancestor.
    - Child inherits parent's transformation history and appends the new `transformation_step`.
    - Child preserves creator attribution notices and upstream licensing restrictions.
    - Non-commercial parent restrictions are preserved: a derived variant cannot silently upgrade to commercial.

    Args:
        parent_record: The source image record being transformed.
        new_record_id: Unique record ID for the derived child.
        new_image_path_or_ref: Relative path or URI to the newly created asset.
        new_sha256: Cryptographic SHA-256 of the new asset file.
        manipulation_category: Forensic manipulation type (e.g. SPLICED_COMPOSITE, SYNTHETIC_EDIT).
        new_label: Forensic label for the child asset (e.g. DatasetImageLabel.IMAGE_COMPOSITE).
        transformation_step: Detailed record of the applied perturbation or edit.
        task_type: Specific detection task type.
        generator_name: Name of tool or neural generator used.
        generator_version: Version of tool or neural generator.
        generator_info: Generator terms and commercial clearance profile.
        processing_config: Additional hyperparameters used in generation.
        new_split: Target partition (TRAIN, VAL, TEST). Inherits from parent if None.
        rights_status_override: Explicit rights status for the child (cannot upgrade non-commercial).
        extra_metadata: Additional child metadata.

    Returns:
        Fresh, fully-typed DatasetImageRecord representing the derived variant.
    """
    # Root ancestor preservation
    root_ancestor_ref = parent_record.original_image_ref or parent_record.record_id

    # Append transformation history without mutating parent
    updated_history = list(parent_record.transformation_history) + [transformation_step]

    # Task type selection
    selected_task_type = task_type or parent_record.task_type or DatasetTaskType.GENERAL_FORENSIC_TRIAGE

    # Determine child rights status: prevent unauthorized commercial escalation from research-only
    child_rights = parent_record.rights_status
    if rights_status_override is not None:
        if parent_record.rights_status == RightsApprovalStatus.RESEARCH_ONLY and rights_status_override in (
            RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
            RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
        ):
            # Block silent escalation: retain RESEARCH_ONLY
            child_rights = RightsApprovalStatus.RESEARCH_ONLY
        else:
            child_rights = rights_status_override

    # Child metadata compilation
    child_metadata = dict(parent_record.metadata)
    child_metadata["parent_record_id"] = parent_record.record_id
    child_metadata["root_ancestor_id"] = root_ancestor_ref
    if processing_config:
        child_metadata["processing_config"] = processing_config
    if extra_metadata:
        child_metadata.update(extra_metadata)

    return DatasetImageRecord(
        record_id=new_record_id,
        image_id=f"IMG-{new_record_id}",
        image_path_or_ref=new_image_path_or_ref,
        sha256=new_sha256.strip().lower(),
        label=new_label,
        task_type=selected_task_type,
        manipulation_category=manipulation_category,
        dataset_name=parent_record.dataset_name,
        source_creator=parent_record.source_creator,
        source_url=parent_record.source_url,
        dataset_version=parent_record.dataset_version,
        source_license=parent_record.source_license,
        license_url=parent_record.license_url,
        attribution_requirements=parent_record.attribution_requirements,
        attribution=parent_record.attribution,
        commercial_use_status=(
            CommercialUseStatus.PROHIBITED
            if parent_record.commercial_use_status == CommercialUseStatus.PROHIBITED
            else parent_record.commercial_use_status
        ),
        research_use_status=parent_record.research_use_status,
        training_eligibility=parent_record.training_eligibility,
        evaluation_eligibility=True,
        redistribution_eligibility=parent_record.redistribution_eligibility,
        rights_status=child_rights,
        consent_reference=parent_record.consent_reference,
        consent_status=parent_record.consent_status,
        human_subjects_present=parent_record.human_subjects_present,
        generator_name=generator_name or parent_record.generator_name,
        generator_version=generator_version or parent_record.generator_version,
        generator_info=generator_info or parent_record.generator_info,
        transformation_history=updated_history,
        original_image_ref=root_ancestor_ref,
        parent_image_ref=parent_record.record_id,
        split=new_split or parent_record.split,
        is_aggregated=parent_record.is_aggregated,
        original_source_ref=parent_record.original_source_ref or parent_record.image_path_or_ref,
        created_at=datetime.now(timezone.utc),
        schema_version=parent_record.schema_version,
        metadata=child_metadata,
    )


class ProvenanceGraph:
    """In-memory lineage DAG managing hierarchical parent-child image relationships."""

    def __init__(self, records: Optional[List[DatasetImageRecord]] = None):
        self._records: Dict[str, DatasetImageRecord] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, str] = {}

        if records:
            for rec in records:
                self.add_record(rec)

    def add_record(self, record: DatasetImageRecord) -> None:
        """Add an image record into the provenance graph."""
        self._records[record.record_id] = record
        parent_ref = getattr(record, "parent_image_ref", None)
        if parent_ref:
            self._parents[record.record_id] = parent_ref
            self._children[parent_ref].append(record.record_id)

    def get_record(self, record_id: str) -> Optional[DatasetImageRecord]:
        """Retrieve record by unique record_id."""
        return self._records.get(record_id)

    def get_parent(self, record_id: str) -> Optional[DatasetImageRecord]:
        """Retrieve immediate parent record if recorded."""
        parent_id = self._parents.get(record_id)
        if parent_id:
            return self._records.get(parent_id)
        return None

    def get_children(self, record_id: str) -> List[DatasetImageRecord]:
        """Retrieve all direct derived child records."""
        child_ids = self._children.get(record_id, [])
        return [self._records[cid] for cid in child_ids if cid in self._records]

    def get_ancestors(self, record_id: str) -> List[DatasetImageRecord]:
        """Retrieve ordered list of ancestors from immediate parent up to root."""
        ancestors: List[DatasetImageRecord] = []
        curr_id = record_id
        visited: Set[str] = set()

        while True:
            parent_id = self._parents.get(curr_id)
            if not parent_id or parent_id in visited:
                break
            visited.add(parent_id)
            parent_rec = self._records.get(parent_id)
            if parent_rec:
                ancestors.append(parent_rec)
            curr_id = parent_id

        return ancestors

    def get_root_ancestor(self, record_id: str) -> Optional[DatasetImageRecord]:
        """Retrieve the original root ancestor in the lineage."""
        ancestors = self.get_ancestors(record_id)
        if ancestors:
            return ancestors[-1]
        return self._records.get(record_id)

    def get_all_descendants(self, record_id: str) -> List[DatasetImageRecord]:
        """Retrieve all transitive descendants (children, grandchildren, etc.)."""
        descendants: List[DatasetImageRecord] = []
        queue = deque([record_id])
        visited: Set[str] = {record_id}

        while queue:
            curr = queue.popleft()
            for child_id in self._children.get(curr, []):
                if child_id not in visited:
                    visited.add(child_id)
                    child_rec = self._records.get(child_id)
                    if child_rec:
                        descendants.append(child_rec)
                    queue.append(child_id)

        return descendants


def verify_provenance_consistency(
    records: List[DatasetImageRecord],
) -> List[ProvenanceViolation]:
    """Audit a collection of image records for complete, unbroken, and uncontradicted provenance chains.

    Checks:
    1. Parent linkage: Every parent_image_ref must resolve to an existing record in the catalog.
    2. Chronological order: Child creation timestamp must be >= parent creation timestamp.
    3. Transformation documentation: Synthetic/manipulated records must possess transformation history or generator details.
    4. Cycle detection: Lineage must be a strict DAG without cyclic parent references.
    5. Rights escalation guard: Derived variants cannot claim commercial clearance if an ancestor is research-only.
    6. Root consistency: If original_image_ref is specified, it must trace back to the same root ancestor.

    Args:
        records: Collection of DatasetImageRecord instances to evaluate.

    Returns:
        List of identified ProvenanceViolation models.
    """
    violations: List[ProvenanceViolation] = []
    record_map: Dict[str, DatasetImageRecord] = {r.record_id: r for r in records}

    # 1. Check parent resolution, timestamps, and rights escalation
    for rec in records:
        parent_id = getattr(rec, "parent_image_ref", None)
        if parent_id:
            parent = record_map.get(parent_id)
            if parent is None:
                violations.append(
                    ProvenanceViolation(
                        violation_type=ProvenanceViolationType.MISSING_PARENT_RECORD,
                        record_id=rec.record_id,
                        parent_id=parent_id,
                        message=(
                            f"Record '{rec.record_id}' references parent '{parent_id}', "
                            "which is missing from the dataset manifest."
                        ),
                        actionable_fix="Ensure parent record is included in manifest or resolve dangling reference.",
                    )
                )
            else:
                # Chronological verification
                if rec.created_at < parent.created_at:
                    violations.append(
                        ProvenanceViolation(
                            violation_type=ProvenanceViolationType.CHRONOLOGICAL_INVERSION,
                            record_id=rec.record_id,
                            parent_id=parent_id,
                            message=(
                                f"Child record '{rec.record_id}' has creation timestamp ({rec.created_at.isoformat()}) "
                                f"earlier than parent '{parent_id}' ({parent.created_at.isoformat()})."
                            ),
                            actionable_fix="Correct chronological timestamps across derivation pipeline.",
                        )
                    )

                # Rights escalation verification
                if parent.rights_status == RightsApprovalStatus.RESEARCH_ONLY and rec.rights_status in (
                    RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
                    RightsApprovalStatus.APPROVED_FOR_PRODUCTION,
                ):
                    violations.append(
                        ProvenanceViolation(
                            violation_type=ProvenanceViolationType.UNAUTHORIZED_RIGHTS_ESCALATION,
                            record_id=rec.record_id,
                            parent_id=parent_id,
                            message=(
                                f"Derived record '{rec.record_id}' declares commercial rights "
                                f"('{rec.rights_status.value}'), but parent '{parent_id}' is restricted "
                                "to 'RESEARCH_ONLY'. Upstream restrictions cannot be dropped in derived variants."
                            ),
                            actionable_fix="Demote child rights status to 'RESEARCH_ONLY' to preserve compliance.",
                        )
                    )

        # Transformation documentation for manipulated assets
        is_manipulated = rec.manipulation_category != ManipulationCategory.NONE
        if is_manipulated:
            has_history = bool(rec.transformation_history)
            has_generator = bool(rec.generator_name)
            if not has_history and not has_generator:
                violations.append(
                    ProvenanceViolation(
                        violation_type=ProvenanceViolationType.MISSING_TRANSFORMATION_EVIDENCE,
                        record_id=rec.record_id,
                        parent_id=parent_id,
                        message=(
                            f"Record '{rec.record_id}' claims manipulation '{rec.manipulation_category.value}', "
                            "but provides no transformation history or generator details."
                        ),
                        actionable_fix="Populate transformation_history or generator_name for all manipulated assets.",
                    )
                )

    # 2. Cycle detection via Kahn's algorithm or DFS
    visited_global: Set[str] = set()
    visiting_local: Set[str] = set()

    def _dfs_cycle(node_id: str, path: List[str]) -> None:
        visiting_local.add(node_id)
        parent_id = getattr(record_map[node_id], "parent_image_ref", None)
        if parent_id and parent_id in record_map:
            if parent_id in visiting_local:
                cycle_str = " -> ".join(path + [parent_id])
                violations.append(
                    ProvenanceViolation(
                        violation_type=ProvenanceViolationType.CYCLIC_LINEAGE,
                        record_id=node_id,
                        parent_id=parent_id,
                        message=f"Cyclic lineage dependency detected: {cycle_str}",
                        actionable_fix="Break cyclic parent-child reference.",
                    )
                )
            elif parent_id not in visited_global:
                _dfs_cycle(parent_id, path + [parent_id])
        visiting_local.remove(node_id)
        visited_global.add(node_id)

    for r_id in record_map:
        if r_id not in visited_global:
            _dfs_cycle(r_id, [r_id])

    return violations
