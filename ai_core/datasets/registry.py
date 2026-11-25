"""Dataset registry for SafeHER image dataset pipeline.

Maintains registered candidate datasets across two distinct compliance tiers:
1. Research-only or restricted datasets:
   - GenImage
   - FaceForensics++
   - DeeperForensics-1.0
   - DFDC (pending agreement review)
   - Explainable Deepfake Detection Challenge 2026

2. Commercial candidates requiring provenance review:
   - Real vs AI Corpus
   - Open Images
   - First-party consent-based photographs
   - Commercially permitted synthetic images and edits

All unverified records in the initial registry are marked NOT_APPROVED_FOR_PRODUCTION.
No external images or large weights are downloaded.
"""

from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.manifest import (
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
)


class CandidateCategory(str, Enum):
    """Categorization of dataset candidate according to technical/legal restrictions."""

    RESEARCH_RESTRICTED = "research_restricted"
    COMMERCIAL_CANDIDATE_REQUIRING_PROVENANCE_REVIEW = "commercial_candidate_requiring_provenance_review"


class CandidateDatasetProfile(BaseModel):
    """Profile specification for a registered candidate dataset."""

    dataset_id: str = Field(..., description="Unique machine-readable dataset identifier")
    name: str = Field(..., description="Official title of dataset")
    category: CandidateCategory = Field(..., description="Compliance classification category")
    description: str = Field(..., description="Forensic and technical summary")
    source_creator: str = Field(..., description="Originating research group, provider, or institution")
    source_url: str = Field(..., description="Official project or documentation URL")
    reviewed_version: str = Field(..., description="Exact dataset or benchmark version audited")
    review_date: str = Field(..., description="ISO date of the latest legal and provenance review")
    primary_license: str = Field(..., description="Primary upstream license identifier")
    license_url: Optional[str] = Field(None, description="Direct link to license agreement")
    terms_of_use_summary: str = Field(..., description="Summary of key contractual terms of use")
    upstream_restrictions: str = Field(..., description="Upstream copyright, license, or origin constraints")
    derivative_work_restrictions: str = Field(..., description="Restrictions on model weights or derivative works")
    commercial_training_permitted: bool = Field(..., description="Whether commercial model training is legally permitted")
    weight_redistribution_conditions: str = Field(..., description="Conditions for distributing models trained on this data")
    consent_withdrawal_policy: str = Field(..., description="Policy or mechanism regarding subject consent and data withdrawal")
    is_aggregated: bool = Field(..., description="Whether dataset is compiled from heterogeneous upstream sources")
    human_subjects_involved: bool = Field(..., description="Whether human subjects or facial likenesses are present")
    rights_status: RightsApprovalStatus = Field(
        default=RightsApprovalStatus.UNVERIFIED,
        description="Current technical rights approval status",
    )
    default_rights_status: RightsApprovalStatus = Field(
        default=RightsApprovalStatus.UNVERIFIED,
        description="Initial technical rights status (backward compatibility alias)",
    )
    allowed_usage: List[UsageScope] = Field(
        ...,
        description="Technically authorized usage scopes",
    )
    attribution_required: bool = Field(
        default=False,
        description="Whether attribution is required by upstream license",
    )
    default_attribution_text: Optional[str] = Field(
        None,
        description="Default attribution string if required",
    )
    unresolved_approval_requirements: List[str] = Field(
        default_factory=list,
        description="Outstanding legal/technical review requirements before production eligibility",
    )
    technical_notes: str = Field(
        ...,
        description="Specific forensic characteristics, generator coverage, and domain constraints",
    )

    model_config = ConfigDict(frozen=True)


DATASET_REGISTRY: Dict[str, CandidateDatasetProfile] = {
    # --------------------------------------------------------------------------
    # Research-only or restricted candidates
    # --------------------------------------------------------------------------
    "genimage": CandidateDatasetProfile(
        dataset_id="genimage",
        name="GenImage",
        category=CandidateCategory.RESEARCH_RESTRICTED,
        description=(
            "Large-scale benchmark covering 8 generative AI architectures (Stable Diffusion v1.4/v1.5, "
            "Midjourney v5, DALL-E 2, GLIDE, VQDM, Wukong, ADM, BigGAN) paired with ImageNet authentic samples."
        ),
        source_creator="Zhu et al. / Shanghai Jiao Tong University & Academic Consortium",
        source_url="https://github.com/GenImage-Dataset/GenImage",
        reviewed_version="v1.0 (NeurIPS 2023)",
        review_date="2026-11-24",
        primary_license="GenImage-Academic-License",
        license_url="https://github.com/GenImage-Dataset/GenImage#license",
        terms_of_use_summary=(
            "Academic research use only. The creators explicitly disclaim commercial liability and "
            "restrict usage strictly to scientific research, benchmarking, and educational evaluation."
        ),
        upstream_restrictions=(
            "Authentic subset is derived directly from the ImageNet 2012 validation set. ImageNet's "
            "upstream terms strictly prohibit commercial use and commercial redistribution. Synthetic subset "
            "includes unconsented depictions and copyrighted concepts generated by commercial and open diffusion models."
        ),
        derivative_work_restrictions=(
            "Model weights trained on GenImage are encumbered by upstream ImageNet non-commercial restrictions "
            "and cannot be commercialized or embedded into commercial detection software."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Research and academic publication only; commercial distribution of trained weights is legally "
            "prohibited due to upstream ImageNet non-commercial covenants."
        ),
        consent_withdrawal_policy=(
            "No subject consent mechanism provided. Dataset contains scraped web faces and public domain objects "
            "with no individual opt-out or likeness revocation path."
        ),
        is_aggregated=True,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        default_rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="GenImage: A Large-Scale ImageNet-Scale Benchmark for AI-Generated Image Detection (Zhu et al., NeurIPS 2023)",
        unresolved_approval_requirements=[
            "Non-commercial academic upstream terms restrict usage strictly to scientific research and evaluation.",
            "Contains ImageNet derivatives with unconsented third-party facial likenesses.",
            "Exempt from commercial training manifests.",
        ],
        technical_notes="Valuable for cross-generator synthetic artifact evaluation. Purely evaluation/research benchmark.",
    ),
    "faceforensics_pp": CandidateDatasetProfile(
        dataset_id="faceforensics_pp",
        name="FaceForensics++",
        category=CandidateCategory.RESEARCH_RESTRICTED,
        description=(
            "Benchmark dataset containing manipulated facial videos generated by 4 automated methods "
            "(Deepfakes, Face2Face, FaceSwap, NeuralTextures) over 1,000 pristine YouTube sequences."
        ),
        source_creator="Technical University of Munich (TUM) / Roessler et al.",
        source_url="https://github.com/ondyari/FaceForensics",
        reviewed_version="v3.0 / C23/C40/raw (ICCV 2019)",
        review_date="2026-11-24",
        primary_license="FaceForensics-Academic-Agreement",
        license_url="https://github.com/ondyari/FaceForensics/blob/master/README.md",
        terms_of_use_summary=(
            "Strict academic agreement required by TUM. Licensees agree not to use the dataset or its "
            "derivatives for commercial purposes, monetization, or production deployments."
        ),
        upstream_restrictions=(
            "1,000 pristine videos downloaded from YouTube under standard YouTube terms without explicit "
            "biometric likeness or commercial releases from featured content creators."
        ),
        derivative_work_restrictions=(
            "TUM terms prohibit redistribution of dataset assets and derivative commercial models. Models "
            "trained on FF++ must remain academic/research only."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Trained weights may be distributed solely for non-commercial research replication."
        ),
        consent_withdrawal_policy=(
            "TUM honors takedown requests if original YouTube videos are removed, but no proactive subject "
            "consent withdrawal mechanism exists for third parties."
        ),
        is_aggregated=True,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        default_rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="FaceForensics++: Learning to Detect Manipulated Facial Images (Roessler et al., ICCV 2019)",
        unresolved_approval_requirements=[
            "Custom Academic Research Agreement explicitly prohibits commercial use or model weight derivation for production.",
            "YouTube source videos scraped without commercial biometric likeness release from subjects.",
            "Redistribution and production training strictly prohibited.",
        ],
        technical_notes="Standard academic baseline for facial manipulation detection; strictly restricted to research and offline benchmarking.",
    ),
    "deeperforensics_1_0": CandidateDatasetProfile(
        dataset_id="deeperforensics_1_0",
        name="DeeperForensics-1.0",
        category=CandidateCategory.RESEARCH_RESTRICTED,
        description=(
            "Large-scale face swap detection benchmark featuring 100 recruited consenting actors "
            "with controlled camera parameters and real-world post-processing perturbations."
        ),
        source_creator="Nanyang Technological University (NTU) & SenseTime / Jiang et al.",
        source_url="https://github.com/EndlessSora/DeeperForensics-1.0",
        reviewed_version="v1.0 (CVPR 2020)",
        review_date="2026-11-24",
        primary_license="DeeperForensics-Academic-License",
        license_url="https://github.com/EndlessSora/DeeperForensics-1.0#license",
        terms_of_use_summary=(
            "Academic research license granted by NTU/SenseTime. Prohibits any commercial use, "
            "reverse engineering for commercial tools, or commercial model training."
        ),
        upstream_restrictions=(
            "Actor consents were executed under an academic IRB protocol specifically limiting image and "
            "biometric data usage to scientific research."
        ),
        derivative_work_restrictions=(
            "Any commercial product derived from DeeperForensics-1.0 is a breach of the academic agreement."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Model weights may only be shared with verified research institutions for reproducible science."
        ),
        consent_withdrawal_policy=(
            "Subject consent governed by NTU institutional ethics board with academic withdrawal procedures; "
            "commercial consent does not exist."
        ),
        is_aggregated=False,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        default_rights_status=RightsApprovalStatus.RESEARCH_ONLY,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="DeeperForensics-1.0: A Large-Scale Dataset for Face Swapping Detection (Jiang et al., CVPR 2020)",
        unresolved_approval_requirements=[
            "Academic license agreement restricts usage exclusively to non-commercial academic research.",
            "Actor likeness consent granted strictly for research evaluation, not commercial training.",
        ],
        technical_notes="High visual quality face swaps with controlled perturbations. Held-out research benchmark only.",
    ),
    "dfdc": CandidateDatasetProfile(
        dataset_id="dfdc",
        name="DFDC (Deepfake Detection Challenge)",
        category=CandidateCategory.RESEARCH_RESTRICTED,
        description=(
            "Massive deepfake video benchmark consisting of over 100,000 video clips generated with "
            "various face-swap and reenactment algorithms using consenting paid actors."
        ),
        source_creator="Meta / Kaggle / AWS / Partnership on AI",
        source_url="https://ai.meta.com/datasets/dfdc/",
        reviewed_version="Full Release (2020)",
        review_date="2026-11-24",
        primary_license="DFDC-Challenge-Agreement",
        license_url="https://ai.meta.com/datasets/dfdc/",
        terms_of_use_summary=(
            "Governed by the Kaggle DFDC Competition Rules and Meta Dataset Agreement. Non-exclusive evaluation license."
        ),
        upstream_restrictions=(
            "Actors received compensation for participating in challenge data capture. However, participant agreement "
            "terms restrict broad commercial exploitation and sublicensing."
        ),
        derivative_work_restrictions=(
            "Ambiguous terms in Clause 4.b regarding commercialization of derived model weights post-competition. "
            "Requires formal enterprise legal opinion."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Redistribution of raw dataset prohibited; commercial distribution of trained weights is unverified "
            "and legally contested."
        ),
        consent_withdrawal_policy=(
            "Actor consent administered by Meta with formal privacy request channels; no API for real-time "
            "automated revocation synchronization."
        ),
        is_aggregated=False,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.PENDING_REVIEW,
        default_rights_status=RightsApprovalStatus.PENDING_REVIEW,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="The Deepfake Detection Challenge (DFDC) Dataset (Dolhansky et al., arXiv 2020)",
        unresolved_approval_requirements=[
            "Pending formal agreement review regarding permissible commercial enterprise model deployment.",
            "Competition agreement clauses require legal clarification on derivative model weight ownership.",
            "Marked unverified PENDING_REVIEW pending legal review completion.",
        ],
        technical_notes="Large diversity of facial scales, lighting, and compression; blocked from production training pending final legal clearance.",
    ),
    "eddc_2026": CandidateDatasetProfile(
        dataset_id="eddc_2026",
        name="Explainable Deepfake Detection Challenge 2026",
        category=CandidateCategory.RESEARCH_RESTRICTED,
        description=(
            "Benchmarking suite focusing on forensic explainability, spatial localization masks, and "
            "fine-grained attribution across novel 2025/2026 generative diffusion architectures."
        ),
        source_creator="Explainable Deepfake Detection Challenge 2026 Consortium",
        source_url="https://eddc2026.org/challenge",
        reviewed_version="2026 Benchmark v1.0",
        review_date="2026-11-24",
        primary_license="EDDC-2026-Academic",
        license_url="https://eddc2026.org/challenge/terms",
        terms_of_use_summary=(
            "Challenge evaluation terms strictly limit dataset usage to benchmark evaluation, leaderboard verification, "
            "and research methodology validation."
        ),
        upstream_restrictions=(
            "Curated evaluation test sets include held-out generative masks; participants agree not to train models on "
            "test splits or monetize challenge test assets."
        ),
        derivative_work_restrictions=(
            "Commercial training prohibited; models must be evaluated solely for forensic explainability metrics."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Evaluation metrics and explainability heatmaps may be published; raw data or derived commercial products "
            "are prohibited."
        ),
        consent_withdrawal_policy=(
            "Challenge consortium handles dataset integrity and participant terms; no individual human likeness withdrawal "
            "required for synthetic subset."
        ),
        is_aggregated=True,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.APPROVED_FOR_EVALUATION_ONLY,
        default_rights_status=RightsApprovalStatus.APPROVED_FOR_EVALUATION_ONLY,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="Explainable Deepfake Detection Challenge 2026 (EDDC-2026 Consortium)",
        unresolved_approval_requirements=[
            "Challenge evaluation terms restrict dataset usage to non-commercial benchmarking and ranking.",
            "Commercial rights and post-challenge commercial training terms unverified.",
        ],
        technical_notes="Forensic localization ground-truth masks; useful strictly for explainability and localization validation.",
    ),

    # --------------------------------------------------------------------------
    # Commercial candidates requiring provenance review
    # --------------------------------------------------------------------------
    "real_vs_ai": CandidateDatasetProfile(
        dataset_id="real_vs_ai",
        name="Real vs AI Corpus",
        category=CandidateCategory.COMMERCIAL_CANDIDATE_REQUIRING_PROVENANCE_REVIEW,
        description=(
            "Heterogeneous multi-source web crawl aggregating synthetic imagery from popular generative "
            "platforms paired with photographic web captures across various object classes."
        ),
        source_creator="Multi-Source Web Crawl / Community Aggregators",
        source_url="https://huggingface.co/datasets/real-vs-ai",
        reviewed_version="HuggingFace Release v1.2 (2024)",
        review_date="2026-11-24",
        primary_license="Mixed-Upstream-Unverified",
        license_url=None,
        terms_of_use_summary=(
            "No formal unified license; repository hosted on public community hub without warranty, provenance "
            "verification, or commercial guarantees."
        ),
        upstream_restrictions=(
            "Web scraped from unverified sources (Reddit, social media, Flickr, prompt galleries). Many authentic images "
            "are protected by full commercial copyright."
        ),
        derivative_work_restrictions=(
            "Significant legal exposure: unverified copyright and lack of consent will taint any commercial model "
            "trained on this data."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Prohibited for commercial model weights until every individual asset is audited and proven commercially unencumbered."
        ),
        consent_withdrawal_policy="No subject consent mechanism or takedown policy exists from community aggregator.",
        is_aggregated=True,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.UNVERIFIED,
        default_rights_status=RightsApprovalStatus.UNVERIFIED,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="Real vs AI Corpus (Community Multi-Source Compilation)",
        unresolved_approval_requirements=[
            "Requires granular per-image upstream license audit to filter out non-commercial and copyrighted assets.",
            "Web-scraped authentic photography lacks verified commercial copyright releases.",
            "Unverified generator terms for synthetic subsets must be reviewed per family.",
        ],
        technical_notes="Must NOT be used in commercial training without item-by-item license purification and provenance verification.",
    ),
    "open_images": CandidateDatasetProfile(
        dataset_id="open_images",
        name="Open Images",
        category=CandidateCategory.COMMERCIAL_CANDIDATE_REQUIRING_PROVENANCE_REVIEW,
        description=(
            "Large-scale dataset of ~9 million URLs with bounding boxes, segmentation masks, and localized "
            "labels harvested from Flickr under various Creative Commons licenses."
        ),
        source_creator="Google / Flickr Creative Commons Contributors",
        source_url="https://storage.googleapis.com/openimages/web/index.html",
        reviewed_version="V7 (2022)",
        review_date="2026-11-24",
        primary_license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms_of_use_summary=(
            "Google releases annotations under CC-BY-4.0. Google expressly states that images themselves are "
            "subject to the independent licenses chosen by their respective Flickr uploaders."
        ),
        upstream_restrictions=(
            "Individual images are licensed under disparate Flickr CC licenses (CC0, CC-BY-2.0, CC-BY-SA, CC-BY-NC, CC-BY-ND). "
            "CC-BY-NC images are strictly non-commercial and CANNOT be used for commercial training."
        ),
        derivative_work_restrictions=(
            "Models trained on CC-BY-NC images inherit non-commercial taint. Commercial models may only train on "
            "filtered CC0 and CC-BY subsets with compliant attribution."
        ),
        commercial_training_permitted=False,
        weight_redistribution_conditions=(
            "Permitted for models trained on CC0 and CC-BY subsets provided license notices and attribution metadata "
            "are preserved in the model release."
        ),
        consent_withdrawal_policy=(
            "Flickr creators retain the right to delete images or change licensing on Flickr. Pipeline must support "
            "hash/URL revocation lists."
        ),
        is_aggregated=True,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.PENDING_REVIEW,
        default_rights_status=RightsApprovalStatus.PENDING_REVIEW,
        allowed_usage=[UsageScope.RESEARCH, UsageScope.EVALUATION],
        attribution_required=True,
        default_attribution_text="Open Images Dataset V7 (Kuznetsova et al., IJCV 2020) / Original Flickr Creators",
        unresolved_approval_requirements=[
            "Umbrella CC-BY-4.0 license covers annotations only; underlying image assets have disparate Flickr licenses.",
            "Must isolate CC0 and CC-BY individual assets and strictly purge CC-BY-NC and CC-BY-ND items.",
            "Per-image creator attribution metadata must be extracted and populated for CC-BY assets.",
        ],
        technical_notes="Excellent candidate for authentic camera baseline if filtered strictly for CC0/CC-BY assets.",
    ),
    "first_party_consent_photos": CandidateDatasetProfile(
        dataset_id="first_party_consent_photos",
        name="First-Party Consent-Based Photographs",
        category=CandidateCategory.COMMERCIAL_CANDIDATE_REQUIRING_PROVENANCE_REVIEW,
        description=(
            "Proprietary studio and smartphone camera dataset created directly with consented adult participants "
            "under documented commercial likeness and biometric model training releases."
        ),
        source_creator="SafeHER Internal Evidence Studio / Consented Subjects",
        source_url="safeher://registry/internal/first_party_consent",
        reviewed_version="SafeHER Studio & Smartphone Collection v2.0",
        review_date="2026-11-24",
        primary_license="SafeHER-First-Party-Commercial",
        license_url="safeher://legal/agreements/biometric_likeness_release_v2.pdf",
        terms_of_use_summary=(
            "Full proprietary commercial rights granted by participants under explicit, informed biometric and "
            "likeness consent agreements compliant with GDPR Article 9 and Illinois BIPA."
        ),
        upstream_restrictions="None; all assets captured first-party using controlled studio equipment and calibrated smartphone sensors.",
        derivative_work_restrictions=(
            "SafeHER holds all intellectual property and commercial deployment rights for derivative models, "
            "weights, and embedded detector systems."
        ),
        commercial_training_permitted=True,
        weight_redistribution_conditions=(
            "Commercial model weights may be deployed in proprietary SafeHER enterprise products and SaaS APIs."
        ),
        consent_withdrawal_policy=(
            "Cryptographically secured tokenized consent vault. Participants may revoke consent using their private "
            "receipt token. Upon revocation, corresponding image records are purged from training sets."
        ),
        is_aggregated=False,
        human_subjects_involved=True,
        rights_status=RightsApprovalStatus.PENDING_REVIEW,
        default_rights_status=RightsApprovalStatus.PENDING_REVIEW,
        allowed_usage=[
            UsageScope.RESEARCH,
            UsageScope.COMMERCIAL_TRAINING,
            UsageScope.EVALUATION,
            UsageScope.REDISTRIBUTION,
        ],
        attribution_required=False,
        default_attribution_text="SafeHER First-Party Consent Collection (Internal Evidence Studio)",
        unresolved_approval_requirements=[
            "Requires automated verification of protected non-identifying consent vault reference (e.g. CR-SAFEHER-2026-F...).",
            "Formal pre-training audit sign-off by legal and privacy advocate lead before elevating to APPROVED_FOR_COMMERCIAL_TRAINING.",
        ],
        technical_notes="Gold-standard ethical candidate for facial manipulation and authentic camera training; requires verified consent tokens.",
    ),
    "commercial_synthetic_edits": CandidateDatasetProfile(
        dataset_id="commercial_synthetic_edits",
        name="Commercially Permitted Synthetic Images and Edits",
        category=CandidateCategory.COMMERCIAL_CANDIDATE_REQUIRING_PROVENANCE_REVIEW,
        description=(
            "Procedural digital forensics perturbations and generative synthetic edits created using in-house "
            "deterministic algorithms (SafeHER ELA/FFT shaders) and commercially licensed generative tools."
        ),
        source_creator="SafeHER Synthetic Forensics Engine / Cleared Commercial Generative Tools",
        source_url="safeher://registry/synthetic/commercial_edits",
        reviewed_version="SafeHER Procedural Forensics Suite v1.0",
        review_date="2026-11-24",
        primary_license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        terms_of_use_summary=(
            "Permissive open-source license permitting unrestricted commercial use, modification, distribution, and patent grants."
        ),
        upstream_restrictions=(
            "Synthesized purely from procedural mathematical noise, algorithmic splicing, or commercially cleared "
            "generative models (e.g. Adobe Firefly v2, SafeHER in-house shaders) with zero unconsented third-party training data."
        ),
        derivative_work_restrictions=(
            "No copyleft restrictions; derivative model weights can be commercially deployed or open-sourced."
        ),
        commercial_training_permitted=True,
        weight_redistribution_conditions="Unrestricted commercial deployment and weight redistribution under Apache-2.0 license terms.",
        consent_withdrawal_policy="Not applicable; zero human subject biometric data or personal likenesses involved.",
        is_aggregated=False,
        human_subjects_involved=False,
        rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        default_rights_status=RightsApprovalStatus.APPROVED_FOR_COMMERCIAL_TRAINING,
        allowed_usage=[
            UsageScope.RESEARCH,
            UsageScope.COMMERCIAL_TRAINING,
            UsageScope.EVALUATION,
        ],
        attribution_required=True,
        default_attribution_text="SafeHER Synthetic Forensics Generator (Apache-2.0)",
        unresolved_approval_requirements=[
            "Verify that any external generative model terms explicitly permit commercial model weight training.",
            "Verify zero third-party trademark, copyrighted character, or non-consented portrait likeness in generation seeds.",
        ],
        technical_notes="Clean procedural synthetic dataset with zero biometric risk; approved for commercial training.",
    ),
}


def get_dataset_profile(dataset_id: str) -> Optional[CandidateDatasetProfile]:
    """Retrieve profile specification for a registered candidate dataset."""
    return DATASET_REGISTRY.get(dataset_id.strip().lower())


def list_all_candidates() -> List[CandidateDatasetProfile]:
    """Return all 9 registered dataset candidate profiles."""
    return list(DATASET_REGISTRY.values())


def list_research_only_candidates() -> List[CandidateDatasetProfile]:
    """Return research-only or restricted dataset candidates."""
    return [p for p in DATASET_REGISTRY.values() if p.category == CandidateCategory.RESEARCH_RESTRICTED]


def list_commercial_candidates() -> List[CandidateDatasetProfile]:
    """Return commercial candidate datasets requiring provenance review."""
    return [
        p
        for p in DATASET_REGISTRY.values()
        if p.category == CandidateCategory.COMMERCIAL_CANDIDATE_REQUIRING_PROVENANCE_REVIEW
    ]


def create_candidate_sample_record(
    dataset_id: str,
    record_index: int = 1,
    split: DatasetSplit = DatasetSplit.TEST,
) -> DatasetImageRecord:
    """Create a mock/sample DatasetImageRecord representing a typical asset from a registered candidate.

    All samples are initialized with deterministic SHA-256 digests and marked NOT_APPROVED_FOR_PRODUCTION.
    No network downloads or large files are fetched.
    """
    profile = get_dataset_profile(dataset_id)
    if profile is None:
        raise KeyError(f"Dataset '{dataset_id}' not found in registry.")

    # Deterministic mock SHA-256 digest
    seed_str = f"safeher-registry-sample:{profile.dataset_id}:{record_index:04d}"
    mock_hash = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()

    # Determine manipulation category and label
    if dataset_id in ("genimage",):
        category = ManipulationCategory.DIFFUSION_GENERATED
        label = DatasetImageLabel.AI_GENERATED_IMAGE
        task_type = DatasetTaskType.AI_GENERATED_IMAGE_DETECTION
        gen_name = "StableDiffusion-v1.5"
        gen_version = "1.5"
    elif dataset_id in ("faceforensics_pp", "deeperforensics_1_0", "dfdc"):
        category = ManipulationCategory.DEEPFAKE_FACE_SWAP
        label = DatasetImageLabel.FACE_SWAP
        task_type = DatasetTaskType.FACE_MANIPULATION_DETECTION
        gen_name = "FaceSwap-GAN"
        gen_version = "2.0"
    elif dataset_id in ("commercial_synthetic_edits",):
        category = ManipulationCategory.SPLICED_COMPOSITE
        label = DatasetImageLabel.IMAGE_COMPOSITE
        task_type = DatasetTaskType.IMAGE_SPLICING_DETECTION
        gen_name = "SafeHER-Procedural-v1"
        gen_version = "1.0"
    elif dataset_id in ("open_images", "first_party_consent_photos"):
        category = ManipulationCategory.NONE
        label = DatasetImageLabel.CAMERA_PHOTOGRAPH
        task_type = DatasetTaskType.GENERAL_FORENSIC_TRIAGE
        gen_name = None
        gen_version = None
    else:
        category = ManipulationCategory.UNKNOWN
        label = DatasetImageLabel.UNKNOWN
        task_type = DatasetTaskType.GENERAL_FORENSIC_TRIAGE
        gen_name = None
        gen_version = None

    # Attribution
    attribution_meta = None
    if profile.attribution_required:
        attribution_meta = AttributionMetadata(
            required=True,
            creator=profile.source_creator,
            attribution_text=profile.default_attribution_text,
            license_notice_url=profile.license_url,
        )

    # Human subjects & consent
    consent_ref = None
    consent_stat = ConsentStatus.NOT_APPLICABLE
    if profile.human_subjects_involved:
        if dataset_id == "first_party_consent_photos":
            consent_ref = f"CR-SAFEHER-2026-F{record_index:03d}"
            consent_stat = ConsentStatus.UNVERIFIED  # Defaults unverified until formal review
        elif dataset_id in ("deeperforensics_1_0", "dfdc"):
            consent_ref = f"CR-RESEARCH-ACTOR-{record_index:03d}"
            consent_stat = ConsentStatus.RESEARCH_ONLY
        else:
            consent_ref = None
            consent_stat = ConsentStatus.UNVERIFIED

    generator_info = None
    if gen_name:
        is_comm = dataset_id == "commercial_synthetic_edits"
        generator_info = GeneratorInfo(
            name=gen_name,
            version=gen_version,
            commercial_use_permitted=is_comm,
            terms_reference=profile.license_url,
        )

    return DatasetImageRecord(
        record_id=f"SH-{profile.dataset_id.upper()}-{record_index:04d}",
        image_id=f"IMG-{profile.dataset_id.upper()}-{record_index:04d}",
        image_path_or_ref=f"safeher://registry/{profile.dataset_id}/sample_{record_index:04d}.jpg",
        sha256=mock_hash,
        label=label,
        task_type=task_type,
        manipulation_category=category,
        dataset_name=profile.name,
        source_creator=profile.source_creator,
        source_url=profile.source_url,
        dataset_version="1.0.0",
        source_license=profile.primary_license,
        license_url=profile.license_url,
        attribution_requirements=profile.default_attribution_text,
        attribution=attribution_meta,
        commercial_use_status=(
            CommercialUseStatus.PERMITTED
            if UsageScope.COMMERCIAL_TRAINING in profile.allowed_usage
            else CommercialUseStatus.PROHIBITED
        ),
        research_use_status=ResearchUseStatus.PERMITTED,
        training_eligibility=(UsageScope.COMMERCIAL_TRAINING in profile.allowed_usage),
        evaluation_eligibility=True,
        redistribution_eligibility=(UsageScope.REDISTRIBUTION in profile.allowed_usage),
        rights_status=RightsApprovalStatus.NOT_APPROVED_FOR_PRODUCTION,
        consent_reference=consent_ref,
        consent_status=consent_stat,
        human_subjects_present=profile.human_subjects_involved,
        generator_name=gen_name,
        generator_version=gen_version,
        generator_info=generator_info,
        transformation_history=[
            TransformationStep(
                step_name="registry_registration",
                parameters={"catalog_dataset_id": profile.dataset_id},
            )
        ],
        original_image_ref=(
            f"upstream://{profile.dataset_id}/source_{record_index:04d}" if profile.is_aggregated else None
        ),
        split=split,
        is_aggregated=profile.is_aggregated,
        schema_version="1.0.0",
        metadata={"catalog_id": profile.dataset_id, "mock_sample": True},
    )


def create_initial_registry_manifest(
    manifest_id: Optional[str] = None,
    target_use: TargetUsageScope = TargetUsageScope.EVALUATION,
) -> DatasetManifest:
    """Create a master manifest containing initial sample records from all 9 registered candidates.

    All records are marked NOT_APPROVED_FOR_PRODUCTION.
    """
    records: List[DatasetImageRecord] = []
    for candidate_id in DATASET_REGISTRY:
        rec = create_candidate_sample_record(candidate_id, record_index=1, split=DatasetSplit.TEST)
        records.append(rec)

    return DatasetManifest(
        manifest_id=manifest_id or "SH-MAN-INITIAL-REGISTRY-2026",
        dataset_name="SafeHER Milestone 4.1 Candidate Dataset Registry",
        version="1.0.0",
        target_use=target_use,
        is_aggregated=False,
        general_license="Proprietary-Evaluation-Only",
        records=records,
        metadata={
            "description": "Initial SafeHER dataset candidate registry for rights auditing and provenance evaluation.",
            "unverified_status": "All records initialized as NOT_APPROVED_FOR_PRODUCTION.",
        },
    )
