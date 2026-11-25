"""Dataset manifest schema and models for SafeHER image dataset pipeline.

Enforces:
- Cryptographic provenance and SHA-256 image immutability.
- Explicit, non-overlapping forensic label taxonomy.
- Strict license taxonomy, usage tracking, and multi-tier eligibility.
- Protected, non-identifying human-subject consent references (zero PII storage).
- Complete transformation and generator audit history.
- Semantic schema versioning and safe path validation.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Current default schema version
CURRENT_SCHEMA_VERSION = "1.0.0"


class DatasetImageLabel(str, Enum):
    """Explicit, non-overlapping forensic ground truth image labels."""

    CAMERA_PHOTOGRAPH = "CAMERA_PHOTOGRAPH"
    AI_GENERATED_IMAGE = "AI_GENERATED_IMAGE"
    FACE_SWAP = "FACE_SWAP"
    FACIAL_REENACTMENT = "FACIAL_REENACTMENT"
    FACIAL_INPAINTING = "FACIAL_INPAINTING"
    LOCAL_IMAGE_EDIT = "LOCAL_IMAGE_EDIT"
    IMAGE_COMPOSITE = "IMAGE_COMPOSITE"
    UNKNOWN = "UNKNOWN"
    UNVERIFIED = "UNVERIFIED"


# Backward compatibility aliases
class GroundTruthLabel(str, Enum):
    """Binary legacy labels, mapped cleanly to DatasetImageLabel."""

    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"


class DatasetTaskType(str, Enum):
    """Specific forensic evaluation task scope."""

    DEEPFAKE_DETECTION = "DEEPFAKE_DETECTION"
    FACE_MANIPULATION_DETECTION = "FACE_MANIPULATION_DETECTION"
    AI_GENERATED_IMAGE_DETECTION = "AI_GENERATED_IMAGE_DETECTION"
    IMAGE_SPLICING_DETECTION = "IMAGE_SPLICING_DETECTION"
    LOCALIZED_FORGERY_DETECTION = "LOCALIZED_FORGERY_DETECTION"
    GENERAL_FORENSIC_TRIAGE = "GENERAL_FORENSIC_TRIAGE"


class ManipulationCategory(str, Enum):
    """Fine-grained visual manipulation or generation taxonomy."""

    NONE = "none"
    DEEPFAKE_FACE_SWAP = "deepfake_face_swap"
    FACE_REENACTMENT = "face_reenactment"
    DIFFUSION_GENERATED = "diffusion_generated"
    GAN_GENERATED = "gan_generated"
    DIFFUSION_INPAINTING = "diffusion_inpainting"
    SPLICED_COMPOSITE = "spliced_composite"
    LOCALIZED_RETOUCH = "localized_retouch"
    SYNTHETIC_EDIT = "synthetic_edit"
    COPY_MOVE = "copy_move"
    UNKNOWN = "unknown"


class CommercialUseStatus(str, Enum):
    """Commercial-use licensing status."""

    PERMITTED = "PERMITTED"
    PROHIBITED = "PROHIBITED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    UNVERIFIED = "UNVERIFIED"


class ResearchUseStatus(str, Enum):
    """Research-use licensing status."""

    PERMITTED = "PERMITTED"
    RESTRICTED = "RESTRICTED"
    PROHIBITED = "PROHIBITED"


class UsageScope(str, Enum):
    """Permitted technical and legal scopes for dataset usage."""

    RESEARCH = "research"
    COMMERCIAL_TRAINING = "commercial_training"
    EVALUATION = "evaluation"
    REDISTRIBUTION = "redistribution"


class TargetUsageScope(str, Enum):
    """Target workload scope against which manifests are verified."""

    RESEARCH = "research"
    EVALUATION = "evaluation"
    COMMERCIAL_TRAINING = "commercial_training"
    PRODUCTION_TRAINING = "production_training"


class RightsApprovalStatus(str, Enum):
    """Explicit technical rights verification statuses."""

    UNVERIFIED = "UNVERIFIED"
    PENDING_REVIEW = "PENDING_REVIEW"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    APPROVED_FOR_COMMERCIAL_TRAINING = "APPROVED_FOR_COMMERCIAL_TRAINING"
    APPROVED_FOR_EVALUATION_ONLY = "APPROVED_FOR_EVALUATION_ONLY"
    REJECTED = "REJECTED"
    EXPIRED_OR_REQUIRES_REVIEW = "EXPIRED_OR_REQUIRES_REVIEW"

    # Backward compatibility aliases
    NOT_APPROVED_FOR_PRODUCTION = "UNVERIFIED"
    APPROVED_FOR_PRODUCTION = "APPROVED_FOR_COMMERCIAL_TRAINING"


class DatasetSplit(str, Enum):
    """Partitioning split for model training and evaluation."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    INTERNAL_TEST = "internal_test"
    HELD_OUT_GENERATOR_TEST = "held_out_generator_test"
    CROSS_DATASET_TEST = "cross_dataset_test"
    COMPRESSION_ROBUSTNESS_TEST = "compression_robustness_test"
    UNASSIGNED = "unassigned"


class ConsentStatus(str, Enum):
    """Human subject consent verification state."""

    NOT_APPLICABLE = "not_applicable"
    AUTHORIZED = "authorized"
    RESEARCH_ONLY = "research_only"
    UNVERIFIED = "unverified"
    REVOKED = "revoked"


class TransformationStep(BaseModel):
    """Audit log of image transformation, perturbation, or post-processing."""

    step_name: str = Field(..., description="Name of processing step, e.g. 'resize', 'recompression'")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used during transformation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when step was executed",
    )

    model_config = ConfigDict(frozen=True)


class GeneratorInfo(BaseModel):
    """Provenance details for synthetic or AI-assisted image generation."""

    name: str = Field(..., description="Generator model or algorithm name")
    version: Optional[str] = Field(None, description="Generator version string")
    commercial_use_permitted: bool = Field(
        default=False,
        description="Whether upstream generator terms permit commercial training/use of outputs",
    )
    terms_reference: Optional[str] = Field(
        None,
        description="URL or reference to generator commercial license or terms of service",
    )

    model_config = ConfigDict(frozen=True)


class AttributionMetadata(BaseModel):
    """Attribution notice and requirements."""

    required: bool = Field(default=False, description="Whether original license mandates attribution")
    creator: Optional[str] = Field(None, description="Original creator or copyright holder")
    attribution_text: Optional[str] = Field(None, description="Specific attribution notice string")
    license_notice_url: Optional[str] = Field(None, description="Link to copyright or attribution notice")

    model_config = ConfigDict(frozen=True)


# Regex checking for potential PII in consent references (email, phone, SSN)
_PII_PATTERNS = [
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # phone number
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
]


def validate_non_identifying_consent_reference(ref: Optional[str]) -> None:
    """Ensure consent reference is an opaque, non-identifying token without PII.

    Raises:
        ValueError: If consent reference contains identifiable PII.
    """
    if not ref:
        return
    for pattern in _PII_PATTERNS:
        if pattern.search(ref):
            raise ValueError(
                "Consent reference must be an opaque, non-identifying token (e.g. 'CR-SAFEHER-2026-F001'). "
                "Storing sensitive PII (emails, phone numbers, identity documents) is strictly prohibited."
            )


def validate_dataset_path_or_ref(path_str: str) -> str:
    """Validate relative path or approved dataset reference.

    Rejects directory traversal, null bytes, and dangerous path formats.

    Raises:
        ValueError: If path contains path traversal patterns or invalid characters.
    """
    if not path_str or not path_str.strip():
        raise ValueError("Image path or dataset reference must not be empty.")

    clean = path_str.strip()

    # Reject null bytes
    if "\x00" in clean:
        raise ValueError("Image path contains prohibited null byte.")

    # Reject path traversal (e.g. ../ or ..\)
    if ".." in clean.replace("\\", "/").split("/"):
        raise ValueError(f"Path traversal detected in image reference: '{clean}'.")
    if clean.startswith("../") or clean.startswith("..\\") or "/../" in clean or "\\..\\" in clean:
        raise ValueError(f"Path traversal detected in image reference: '{clean}'.")

    # Approved URI schemes
    approved_schemes = (
        "safeher://",
        "upstream://",
        "dataset://",
        "faceforensics://",
        "genimage://",
        "openimages://",
        "s3://",
        "gs://",
        "https://",
        "http://",
    )
    if any(clean.startswith(scheme) for scheme in approved_schemes):
        return clean

    # If relative filesystem path, ensure it does not start with drive letter or leading root slash
    if re.match(r"^[a-zA-Z]:", clean):
        raise ValueError(f"Absolute Windows drive paths are not permitted in dataset manifests: '{clean}'. Use relative paths or approved dataset references.")
    if clean.startswith("/") or clean.startswith("\\"):
        raise ValueError(f"Absolute filesystem paths are not permitted in dataset manifests: '{clean}'. Use relative paths or approved dataset references.")

    return clean


def compute_sha256(data: Union[bytes, str, Path]) -> str:
    """Compute SHA-256 hex digest for bytes, text, or file path."""
    hasher = hashlib.sha256()
    if isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, Path) or (isinstance(data, str) and Path(data).is_file()):
        with open(data, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(data, str):
        hasher.update(data.encode("utf-8"))
    else:
        raise TypeError(f"Unsupported data type for SHA-256 calculation: {type(data)}")
    return hasher.hexdigest().lower()


class DatasetImageRecord(BaseModel):
    """Manifest record representing a single image asset with complete legal, forensic, and rights provenance."""

    # 1. Unique record ID
    record_id: str = Field(
        default_factory=lambda: f"SH-REC-{uuid4().hex[:12].upper()}",
        description="Unique record identifier",
    )

    # 2. Image identifier
    image_id: str = Field(
        default_factory=lambda: f"IMG-{uuid4().hex[:12].upper()}",
        description="Unique image asset identifier",
    )

    # 3. Relative path or approved dataset reference
    image_path_or_ref: str = Field(
        ...,
        description="Relative path or approved dataset reference (e.g. 'genimage://sdv14/train/001.jpg', 'images/001.jpg')",
    )

    # 4. SHA-256
    sha256: str = Field(
        ...,
        description="64-character lowercase SHA-256 cryptographic digest of image bytes",
    )

    # 5. Label (Explicit non-overlapping label from taxonomy)
    label: DatasetImageLabel = Field(
        ...,
        description="Explicit non-overlapping forensic label (e.g. CAMERA_PHOTOGRAPH, AI_GENERATED_IMAGE, FACE_SWAP)",
    )

    # 6. Task type
    task_type: DatasetTaskType = Field(
        default=DatasetTaskType.GENERAL_FORENSIC_TRIAGE,
        description="Forensic computer vision evaluation task type",
    )

    # 7. Manipulation category
    manipulation_category: ManipulationCategory = Field(
        default=ManipulationCategory.NONE,
        description="Fine-grained category of manipulation or generation technique",
    )

    # 8. Original image reference
    original_image_ref: Optional[str] = Field(
        None,
        description="Reference, URI, or identifier of original unmanipulated source image",
    )

    # 9. Parent image reference
    parent_image_ref: Optional[str] = Field(
        None,
        description="Reference to parent background, donor frame, or source video asset",
    )

    # 10. Dataset name
    dataset_name: str = Field(
        ...,
        description="Source dataset name (e.g. 'FaceForensics++', 'Open Images', 'SafeHER-Internal')",
    )

    # 11. Original creator or provider
    source_creator: str = Field(
        ...,
        description="Originating institution, author, or dataset provider",
    )

    # 12. Source URL
    source_url: Optional[str] = Field(
        None,
        description="URL to dataset paper, repository, or official landing page",
    )

    # 13. Dataset version
    dataset_version: str = Field(
        default="1.0.0",
        description="Version string of source dataset release",
    )

    # 14. Source licence identifier
    source_license: str = Field(
        ...,
        description="SPDX license ID or standard license identifier (e.g. 'Apache-2.0', 'CC-BY-NC-4.0')",
    )

    # 15. Licence URL
    license_url: Optional[str] = Field(
        None,
        description="Direct URL to original license text or agreement",
    )

    # 16. Attribution requirements
    attribution_requirements: Optional[str] = Field(
        None,
        description="Textual description of required attribution, or notice string",
    )
    attribution: Optional[AttributionMetadata] = Field(
        None,
        description="Structured attribution metadata",
    )

    # 17. Commercial-use status
    commercial_use_status: CommercialUseStatus = Field(
        default=CommercialUseStatus.UNVERIFIED,
        description="Commercial usage status according to upstream terms",
    )

    # 18. Research-use status
    research_use_status: ResearchUseStatus = Field(
        default=ResearchUseStatus.PERMITTED,
        description="Academic/research usage status according to upstream terms",
    )

    # 19. Training eligibility
    training_eligibility: bool = Field(
        default=False,
        description="Whether record is technically and legally eligible for model training",
    )

    # 20. Evaluation eligibility
    evaluation_eligibility: bool = Field(
        default=True,
        description="Whether record is eligible for evaluation and benchmarking",
    )

    # 21. Redistribution eligibility
    redistribution_eligibility: bool = Field(
        default=False,
        description="Whether dataset record may be redistributed to third parties",
    )

    # 22. Rights-review status
    rights_status: RightsApprovalStatus = Field(
        default=RightsApprovalStatus.NOT_APPROVED_FOR_PRODUCTION,
        description="Technical rights verification status",
    )

    # 23. Consent reference where required
    consent_reference: Optional[str] = Field(
        None,
        description="Protected non-identifying consent vault reference (e.g. 'CR-SAFEHER-2026-F001')",
    )
    consent_status: ConsentStatus = Field(
        default=ConsentStatus.NOT_APPLICABLE,
        description="Human subject consent verification status",
    )
    human_subjects_present: bool = Field(
        default=False,
        description="Whether human subjects / identifiable faces are present in image",
    )

    # 24. Generator name and version
    generator_name: Optional[str] = Field(
        None,
        description="Name of synthetic image generator or neural model",
    )
    generator_version: Optional[str] = Field(
        None,
        description="Version string of generator model",
    )
    generator_info: Optional[GeneratorInfo] = Field(
        None,
        description="Structured generator licensing details",
    )

    # 25. Transformation history
    transformation_history: List[TransformationStep] = Field(
        default_factory=list,
        description="Chronological record of perturbations, compressions, or processing steps",
    )

    # 26. Data split
    split: DatasetSplit = Field(
        default=DatasetSplit.UNASSIGNED,
        description="Dataset partition split (train, val, test, unassigned)",
    )

    # 27. Grouping identifier
    grouping_id: Optional[str] = Field(
        None,
        description="Grouping identifier (subject ID, video sequence ID, scene ID) to prevent leakage across splits",
    )

    # Aggregated source indicator
    is_aggregated: bool = Field(
        default=False,
        description="Whether this asset originates from an aggregated multi-source dataset",
    )

    # 28. Creation timestamp
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of record creation",
    )

    # 29. Record schema version
    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="Semantic version of record schema (e.g. '1.0.0')",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional technical provenance parameters (resolution, compression, etc.)",
    )

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Backward compatibility properties & accessors
    @property
    def source_dataset(self) -> str:
        """Backward compatibility alias for dataset_name."""
        return self.dataset_name

    @property
    def original_license(self) -> str:
        """Backward compatibility alias for source_license."""
        return self.source_license

    @property
    def original_source_ref(self) -> Optional[str]:
        """Backward compatibility alias for original_image_ref."""
        return self.original_image_ref

    @property
    def allowed_usage(self) -> List[UsageScope]:
        """Derived allowed usage scopes for backward compatibility."""
        scopes: List[UsageScope] = []
        if self.research_use_status == ResearchUseStatus.PERMITTED:
            scopes.append(UsageScope.RESEARCH)
        if self.evaluation_eligibility:
            scopes.append(UsageScope.EVALUATION)
        if self.training_eligibility and self.commercial_use_status == CommercialUseStatus.PERMITTED:
            scopes.append(UsageScope.COMMERCIAL_TRAINING)
        if self.redistribution_eligibility:
            scopes.append(UsageScope.REDISTRIBUTION)
        return scopes

    @model_validator(mode="before")
    @classmethod
    def _handle_backward_compatibility(cls, data: Any) -> Any:
        """Map legacy attribute names (source_dataset, original_license, original_source_ref) into new schema."""
        if isinstance(data, dict):
            # source_dataset -> dataset_name
            if "source_dataset" in data and "dataset_name" not in data:
                data["dataset_name"] = data["source_dataset"]
            elif "dataset_name" in data and "source_dataset" not in data:
                data["source_dataset"] = data["dataset_name"]

            # original_license -> source_license
            if "original_license" in data and "source_license" not in data:
                data["source_license"] = data["original_license"]
            elif "source_license" in data and "original_license" not in data:
                data["original_license"] = data["source_license"]

            # original_source_ref -> original_image_ref
            if "original_source_ref" in data and "original_image_ref" not in data:
                data["original_image_ref"] = data["original_source_ref"]
            elif "original_image_ref" in data and "original_source_ref" not in data:
                data["original_source_ref"] = data["original_image_ref"]

            # label mapping from legacy GroundTruthLabel / strings
            lbl = data.get("label")
            if lbl is not None:
                if lbl in ("authentic", GroundTruthLabel.AUTHENTIC):
                    data["label"] = DatasetImageLabel.CAMERA_PHOTOGRAPH
                elif lbl in ("manipulated", GroundTruthLabel.MANIPULATED):
                    # Infer specific manipulation if present
                    cat = data.get("manipulation_category")
                    if cat == ManipulationCategory.DEEPFAKE_FACE_SWAP:
                        data["label"] = DatasetImageLabel.FACE_SWAP
                    elif cat == ManipulationCategory.FACE_REENACTMENT:
                        data["label"] = DatasetImageLabel.FACIAL_REENACTMENT
                    elif cat == ManipulationCategory.DIFFUSION_GENERATED:
                        data["label"] = DatasetImageLabel.AI_GENERATED_IMAGE
                    elif cat == ManipulationCategory.DIFFUSION_INPAINTING:
                        data["label"] = DatasetImageLabel.FACIAL_INPAINTING
                    elif cat == ManipulationCategory.LOCALIZED_RETOUCH:
                        data["label"] = DatasetImageLabel.LOCAL_IMAGE_EDIT
                    else:
                        data["label"] = DatasetImageLabel.IMAGE_COMPOSITE

            # rights_status legacy mapping
            rs = data.get("rights_status")
            if rs == "NOT_APPROVED_FOR_PRODUCTION":
                data["rights_status"] = RightsApprovalStatus.UNVERIFIED
            elif rs == "APPROVED_FOR_PRODUCTION":
                data["rights_status"] = RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING

            # Sync allowed_usage if passed explicitly
            allowed = data.get("allowed_usage")
            if allowed and isinstance(allowed, list):
                if UsageScope.COMMERCIAL_TRAINING in allowed or "commercial_training" in allowed:
                    data.setdefault("commercial_use_status", CommercialUseStatus.PERMITTED)
                    data.setdefault("training_eligibility", True)
                if UsageScope.RESEARCH in allowed or "research" in allowed:
                    data.setdefault("research_use_status", ResearchUseStatus.PERMITTED)
                if UsageScope.EVALUATION in allowed or "evaluation" in allowed:
                    data.setdefault("evaluation_eligibility", True)
                if UsageScope.REDISTRIBUTION in allowed or "redistribution" in allowed:
                    data.setdefault("redistribution_eligibility", True)

        return data

    @field_validator("sha256")
    @classmethod
    def _validate_sha256_format(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("SHA-256 hash must not be empty.")
        clean = v.strip().lower()
        if len(clean) != 64 or not re.match(r"^[0-9a-f]{64}$", clean):
            raise ValueError(f"Invalid SHA-256 digest format: '{v}'. Expected 64-char hex string.")
        return clean

    @field_validator("record_id", "image_id", "dataset_name", "source_creator", "source_license")
    @classmethod
    def _validate_mandatory_strings(cls, v: str, info) -> str:
        field_name = info.field_name
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError(f"Mandatory provenance field '{field_name}' must not be empty.")
        return v.strip()

    @field_validator("image_path_or_ref")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return validate_dataset_path_or_ref(v)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Schema version must not be empty.")
        clean = v.strip()
        if not re.match(r"^\d+\.\d+\.\d+$", clean):
            raise ValueError(f"Invalid schema version format: '{v}'. Expected semantic versioning (e.g. '1.0.0').")
        return clean

    @field_validator("consent_reference")
    @classmethod
    def _validate_consent_pii(cls, v: Optional[str]) -> Optional[str]:
        validate_non_identifying_consent_reference(v)
        return v.strip() if v else None

    @field_validator("grouping_id")
    @classmethod
    def _validate_grouping_id(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        validate_non_identifying_consent_reference(v)
        clean = v.strip()
        # Prohibit raw biometric embedding vectors in public metadata
        if clean.startswith("[") and clean.endswith("]") and "," in clean:
            # Check if looks like a float embedding array
            tokens = [t.strip() for t in clean[1:-1].split(",")]
            if len(tokens) >= 4 and all(re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", t) for t in tokens[:4]):
                raise ValueError(
                    "Storing raw biometric face embeddings in public grouping_id metadata is strictly prohibited. "
                    "Use an opaque, non-identifying group token (e.g. 'GRP-SUBJECT-001')."
                )
        return clean


class DatasetManifest(BaseModel):
    """Container manifest managing a partitioned collection of image records."""

    manifest_id: str = Field(
        default_factory=lambda: f"SH-MAN-{uuid4().hex[:12].upper()}",
        description="Unique identifier for manifest",
    )
    dataset_name: str = Field(..., description="Dataset or benchmark title")
    version: str = Field(default="1.0.0", description="Semantic version of manifest")
    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="Semantic version of manifest schema",
    )
    target_use: TargetUsageScope = Field(
        default=TargetUsageScope.EVALUATION,
        description="Target workload for which manifest is compiled",
    )
    is_aggregated: bool = Field(
        default=False,
        description="Whether this dataset aggregates images from multiple heterogeneous upstream sources",
    )
    general_license: Optional[str] = Field(
        None,
        description="General license declared by dataset distributor (cannot override upstream restrictions)",
    )
    records: List[DatasetImageRecord] = Field(
        default_factory=list,
        description="List of verified image records",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional manifest metadata",
    )

    model_config = ConfigDict(frozen=True)

    def add_record(self, record: DatasetImageRecord) -> "DatasetManifest":
        """Return a new DatasetManifest with appended record."""
        return self.model_copy(update={"records": self.records + [record]})

    def get_record_by_hash(self, sha256_hash: str) -> Optional[DatasetImageRecord]:
        """Find record by SHA-256 hash."""
        clean = sha256_hash.strip().lower()
        for r in self.records:
            if r.sha256 == clean:
                return r
        return None

    def to_json(self, indent: int = 2) -> str:
        """Export manifest to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        """Deserialize manifest from JSON string."""
        return cls.model_validate_json(json_str)

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist manifest to local filesystem."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "DatasetManifest":
        """Load manifest from local filesystem."""
        path = Path(file_path)
        return cls.from_json(path.read_text(encoding="utf-8"))
