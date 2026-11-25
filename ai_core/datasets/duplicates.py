"""Duplicate detection and perceptual hashing engine for SafeHER image datasets.

Implements:
1. Exact cryptographic duplicate detection using SHA-256 digests.
2. Near-duplicate candidate detection using Difference Hash (dHash) and DCT-based Perceptual Hash (pHash).
3. Configurable similarity thresholds (Hamming distance and percentage similarity).
4. Collision handling and provenance conflict detection.
5. Structured human review flags with strict biometric non-equivalence guarantee:
   Perceptual similarity measures pixel structure; it DOES NOT prove that two images depict the same person.
"""

from collections import defaultdict
import io
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.manifest import DatasetImageRecord


class ExactDuplicateCluster(BaseModel):
    """Cluster of records sharing an identical cryptographic SHA-256 hash."""

    sha256: str = Field(..., description="Cryptographic SHA-256 digest")
    record_ids: List[str] = Field(..., description="IDs of all records in this exact duplicate group")
    count: int = Field(..., ge=2, description="Number of duplicate occurrences")
    has_provenance_conflict: bool = Field(
        ...,
        description="Whether records in this cluster claim contradictory labels, licenses, or sources",
    )
    conflicting_fields: List[str] = Field(
        default_factory=list,
        description="Names of metadata fields that disagree across duplicates",
    )

    model_config = ConfigDict(frozen=True)


class NearDuplicateCandidate(BaseModel):
    """Pair of image assets flagged as potential visual near-duplicates via perceptual hashing."""

    record_id_a: str = Field(..., description="First record identifier")
    record_id_b: str = Field(..., description="Second record identifier")
    hash_a: str = Field(..., description="Hex perceptual hash for record A")
    hash_b: str = Field(..., description="Hex perceptual hash for record B")
    hamming_distance: int = Field(..., ge=0, le=64, description="Bitwise Hamming distance (0-64)")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (1.0 = identical perceptual hash)")
    category: str = Field(..., description="Classification (e.g. 'HIGH_CONFIDENCE_VARIANT', 'STRUCTURAL_SIMILARITY')")
    requires_human_review: bool = Field(
        default=True,
        description="Flag indicating this candidate pair requires manual forensic verification",
    )
    notes: str = Field(..., description="Forensic observations (e.g. recompression, metadata strip, resize)")
    disclaimer: str = Field(
        default=(
            "SafeHER Biometric Non-Equivalence Notice: Perceptual hash similarity measures image-space "
            "structural frequency and pixel distribution. It does NOT constitute facial recognition or "
            "proof that two images depict the same human subject."
        ),
        description="Mandatory disclaimer preventing automated biometric likeness assumptions",
    )

    model_config = ConfigDict(frozen=True)


class DuplicateDetectionReport(BaseModel):
    """Comprehensive duplicate detection report covering exact and near-duplicate assets."""

    total_records_analyzed: int = Field(..., ge=0)
    exact_duplicate_clusters: List[ExactDuplicateCluster] = Field(default_factory=list)
    near_duplicate_candidates: List[NearDuplicateCandidate] = Field(default_factory=list)
    total_exact_duplicate_records: int = Field(default=0)
    total_provenance_conflicts: int = Field(default=0)
    total_near_duplicate_pairs: int = Field(default=0)
    summary: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


def compute_dhash(image_input: Union[bytes, Image.Image], hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for an image.

    dHash tracks relative horizontal luminance gradients. It is extremely fast,
    deterministic, and invariant to uniform scaling, aspect-ratio stretching, and
    metadata alterations.

    Args:
        image_input: Raw encoded image bytes or PIL Image instance.
        hash_size: Grid width/height; default 8 produces a (8x8) = 64-bit hash.

    Returns:
        16-character lowercase hexadecimal hash representing 64 bits.
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        img = image_input

    # Convert to grayscale and resize to (hash_size + 1, hash_size)
    # 9x8 pixels allows 8 horizontal difference comparisons per row
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.int32)

    # Compute horizontal difference: arr[:, 1:] > arr[:, :-1]
    diff = arr[:, 1:] > arr[:, :-1]

    # Convert 64 boolean bits to integer
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_string, 2)

    # Format as 16-character hex string (zero-padded)
    return f"{int_val:016x}"


def compute_phash(image_input: Union[bytes, Image.Image], hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Compute Discrete Cosine Transform (DCT) based perceptual hash (pHash).

    pHash focuses on low-frequency DCT components, providing robustness against
    high-frequency noise, mild re-compression, and minor localized filtering.

    Args:
        image_input: Raw image bytes or PIL Image instance.
        hash_size: Target bit dimension (8 yields 64-bit hash).
        highfreq_factor: Scaling factor for initial DCT calculation.

    Returns:
        16-character lowercase hexadecimal hash.
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        img = image_input

    target_dim = hash_size * highfreq_factor  # 32x32
    resized = img.convert("L").resize((target_dim, target_dim), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)

    # 2D DCT using separable 1D DCTs
    def _dct_1d(a: np.ndarray) -> np.ndarray:
        n = a.shape[-1]
        k = np.arange(n)[:, None]
        m = np.arange(n)[None, :]
        factor = np.cos(np.pi / (2 * n) * (2 * m + 1) * k)
        weights = np.empty(n)
        weights[0] = 1.0 / np.sqrt(n)
        weights[1:] = np.sqrt(2.0 / n)
        return weights[:, None] * (factor @ a.T)

    # Apply 2D DCT: columns then rows
    dct_2d = _dct_1d(_dct_1d(pixels).T).T

    # Extract top-left low frequencies (8x8), excluding DC component at [0, 0]
    low_freq = dct_2d[:hash_size, :hash_size]
    median_val = np.median(low_freq[1:, 1:])  # exclude DC term from median calculation

    # Compute binary hash based on whether each frequency is above median
    bit_array = (low_freq > median_val).flatten()
    bit_string = "".join("1" if b else "0" for b in bit_array)
    int_val = int(bit_string, 2)

    return f"{int_val:016x}"


def hamming_distance(hex_hash_1: str, hex_hash_2: str) -> int:
    """Calculate bitwise Hamming distance between two 64-bit hexadecimal hashes.

    Returns:
        Integer count of differing bits (0 to 64).
    """
    val1 = int(hex_hash_1, 16)
    val2 = int(hex_hash_2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")


def perceptual_similarity(hex_hash_1: str, hex_hash_2: str) -> float:
    """Compute normalized visual similarity score between two 64-bit perceptual hashes.

    Returns:
        Float between 0.0 (completely dissimilar) and 1.0 (identical perceptual hash).
    """
    dist = hamming_distance(hex_hash_1, hex_hash_2)
    return round(1.0 - (dist / 64.0), 4)


class DuplicateDetector:
    """Engine executing exact cryptographic matching and perceptual near-duplicate analysis."""

    def __init__(
        self,
        max_hamming_distance: int = 6,
        hash_strategy: str = "dhash",
    ):
        """Initialize detector with configurable similarity threshold.

        Args:
            max_hamming_distance: Maximum bit difference to flag as near-duplicate (default 6 bits = ~90.6% similarity).
            hash_strategy: Perceptual hash algorithm ('dhash' or 'phash').
        """
        self.max_hamming_distance = max_hamming_distance
        self.hash_strategy = hash_strategy.lower()

    def find_exact_duplicates(
        self,
        records: List[DatasetImageRecord],
    ) -> List[ExactDuplicateCluster]:
        """Detect identical files sharing cryptographic SHA-256 digests and audit for provenance conflicts."""
        hash_groups: Dict[str, List[DatasetImageRecord]] = defaultdict(list)
        for rec in records:
            clean_hash = (rec.sha256 or "").strip().lower()
            if clean_hash:
                hash_groups[clean_hash].append(rec)

        clusters: List[ExactDuplicateCluster] = []
        for h, group in hash_groups.items():
            if len(group) > 1:
                rec_ids = [r.record_id for r in group]
                # Check for metadata contradictions
                conflicts: List[str] = []
                if len({r.label for r in group}) > 1:
                    conflicts.append("label")
                if len({r.manipulation_category for r in group}) > 1:
                    conflicts.append("manipulation_category")
                if len({r.source_dataset for r in group}) > 1:
                    conflicts.append("source_dataset")
                if len({r.original_license for r in group}) > 1:
                    conflicts.append("original_license")

                clusters.append(
                    ExactDuplicateCluster(
                        sha256=h,
                        record_ids=rec_ids,
                        count=len(group),
                        has_provenance_conflict=bool(conflicts),
                        conflicting_fields=conflicts,
                    )
                )

        return clusters

    def find_near_duplicates(
        self,
        image_hashes: Dict[str, str],
        custom_threshold: Optional[int] = None,
    ) -> List[NearDuplicateCandidate]:
        """Find near-duplicate candidate pairs across a mapping of {record_id: hex_perceptual_hash}.

        Args:
            image_hashes: Dictionary mapping record_id to 16-character hex perceptual hash.
            custom_threshold: Optional override for maximum allowed Hamming distance.

        Returns:
            List of NearDuplicateCandidate records flagged for forensic review.
        """
        threshold = custom_threshold if custom_threshold is not None else self.max_hamming_distance
        candidates: List[NearDuplicateCandidate] = []
        record_ids = list(image_hashes.keys())

        for i in range(len(record_ids)):
            id_a = record_ids[i]
            hash_a = image_hashes[id_a]

            for j in range(i + 1, len(record_ids)):
                id_b = record_ids[j]
                hash_b = image_hashes[id_b]

                dist = hamming_distance(hash_a, hash_b)
                if dist <= threshold:
                    score = perceptual_similarity(hash_a, hash_b)
                    if dist <= 2:
                        cat = "HIGH_CONFIDENCE_RECOMPRESSION_OR_FORMAT_CONVERSION"
                        notes = "Extremely high visual similarity; probable re-encoding, minor resize, or metadata strip."
                    elif dist <= 5:
                        cat = "STRUCTURAL_SIMILARITY_CANDIDATE"
                        notes = "Moderate visual similarity; shared composition, background, or slight geometric edit."
                    else:
                        cat = "BORDERLINE_CANDIDATE_FOR_REVIEW"
                        notes = "Borderline structural match within configured threshold; requires human review."

                    candidates.append(
                        NearDuplicateCandidate(
                            record_id_a=id_a,
                            record_id_b=id_b,
                            hash_a=hash_a,
                            hash_b=hash_b,
                            hamming_distance=dist,
                            similarity_score=score,
                            category=cat,
                            requires_human_review=True,
                            notes=notes,
                        )
                    )

        return candidates

    def analyze_dataset(
        self,
        records: List[DatasetImageRecord],
        image_payloads: Optional[Dict[str, Union[bytes, Image.Image]]] = None,
        cached_hashes: Optional[Dict[str, str]] = None,
    ) -> DuplicateDetectionReport:
        """Run complete duplicate analysis across records: exact SHA-256 + near-duplicate perceptual hashing.

        Args:
            records: Manifest records to analyze.
            image_payloads: Optional mapping of record_id to raw image bytes/PIL image.
            cached_hashes: Optional mapping of record_id to pre-computed perceptual hash.

        Returns:
            Structured DuplicateDetectionReport.
        """
        exact_clusters = self.find_exact_duplicates(records)
        total_exact_records = sum(c.count for c in exact_clusters)
        total_conflicts = sum(1 for c in exact_clusters if c.has_provenance_conflict)

        # Compute or collect perceptual hashes
        p_hashes: Dict[str, str] = {}
        if cached_hashes:
            p_hashes.update(cached_hashes)

        if image_payloads:
            for rec_id, payload in image_payloads.items():
                if rec_id not in p_hashes:
                    if self.hash_strategy == "phash":
                        p_hashes[rec_id] = compute_phash(payload)
                    else:
                        p_hashes[rec_id] = compute_dhash(payload)

        near_candidates = self.find_near_duplicates(p_hashes) if len(p_hashes) > 1 else []

        summary = {
            "records_analyzed": len(records),
            "exact_clusters": len(exact_clusters),
            "exact_duplicate_records": total_exact_records,
            "provenance_conflicts": total_conflicts,
            "near_duplicate_pairs": len(near_candidates),
            "perceptual_strategy": self.hash_strategy,
            "max_hamming_threshold": self.max_hamming_distance,
        }

        return DuplicateDetectionReport(
            total_records_analyzed=len(records),
            exact_duplicate_clusters=exact_clusters,
            near_duplicate_candidates=near_candidates,
            total_exact_duplicate_records=total_exact_records,
            total_provenance_conflicts=total_conflicts,
            total_near_duplicate_pairs=len(near_candidates),
            summary=summary,
        )
