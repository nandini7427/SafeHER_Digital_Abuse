"""Generates a reproducible per-image evaluation CSV across all SafeHER benchmark datasets.

Produces a verifiable audit trail containing:
- sample_id
- source_dataset
- ground_truth_label
- is_manipulated
- raw_model_score (discrete thresholded score)
- raw_ela_variance (continuous analog ELA error variance)
- predicted_label
- is_predicted_manipulated
- processing_eligibility
- uncertainty_state
- confusion_matrix_outcome

Contains zero private user evidence; operates strictly on synthetic scientific benchmarks.
"""

import csv
import io
import os
from pathlib import Path
import shutil
import sys
from typing import List

from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_core.evaluation.cross_dataset_benchmark import (
    CrossDatasetSample,
    build_authentic_stress_dataset,
    build_in_distribution_dataset,
    build_pure_ai_generated_dataset,
    build_unseen_manipulation_dataset,
    build_wild_social_media_dataset,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import BaselineHeuristicRunner
from ai_core.image_intelligence.schemas import AssessmentLabel


def generate_evaluation_csv() -> Path:
    detector = SafeHERImageDetector()
    heuristic_runner = BaselineHeuristicRunner(config=detector.config)

    datasets = [
        ("in_distribution_test", build_in_distribution_dataset()),
        ("wild_social_media", build_wild_social_media_dataset()),
        ("unseen_manipulation", build_unseen_manipulation_dataset()),
        ("pure_ai_generated", build_pure_ai_generated_dataset()),
        ("authentic_stress_controls", build_authentic_stress_dataset()),
    ]

    out_file = Path(__file__).resolve().parent / "per_image_evaluation_results.csv"

    fieldnames = [
        "sample_id",
        "source_dataset",
        "ground_truth_label",
        "is_manipulated",
        "raw_model_score",
        "raw_ela_variance",
        "predicted_label",
        "is_predicted_manipulated",
        "processing_eligibility",
        "uncertainty_state",
        "confusion_matrix_outcome",
    ]

    rows = []
    total_samples = 0

    for ds_name, samples in datasets:
        for s in samples:
            total_samples += 1
            resp = detector.detect(s.image_bytes)
            is_pred_manip = resp.assessment_label in (
                AssessmentLabel.SUSPICIOUS_MANIPULATION,
                AssessmentLabel.HIGHLY_LIKELY_MANIPULATED,
            )

            # Continuous analog ELA variance
            img = Image.open(io.BytesIO(s.image_bytes)).convert("RGB")
            _, _, var_diff = heuristic_runner._compute_ela(img)

            # Assign confusion matrix bucket
            if not resp.analysis_eligibility:
                outcome = "UNASSESSED_OOD"
            elif s.is_manipulated and is_pred_manip:
                outcome = "TP"
            elif not s.is_manipulated and is_pred_manip:
                outcome = "FP"
            elif not s.is_manipulated and not is_pred_manip:
                outcome = "TN"
            elif s.is_manipulated and not is_pred_manip:
                outcome = "FN"
            else:
                outcome = "UNKNOWN"

            rows.append(
                {
                    "sample_id": s.sample_id,
                    "source_dataset": ds_name,
                    "ground_truth_label": "manipulated" if s.is_manipulated else "authentic",
                    "is_manipulated": str(s.is_manipulated),
                    "raw_model_score": f"{resp.raw_model_score:.4f}",
                    "raw_ela_variance": f"{var_diff:.4f}",
                    "predicted_label": resp.assessment_label.value,
                    "is_predicted_manipulated": str(is_pred_manip),
                    "processing_eligibility": str(resp.analysis_eligibility),
                    "uncertainty_state": resp.uncertainty_state.value,
                    "confusion_matrix_outcome": outcome,
                }
            )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated per-image evaluation CSV with {len(rows)} samples: {out_file}")

    # Also copy to optional artifact directory if configured
    env_artifact_dir = os.environ.get("SAFEHER_ARTIFACT_DIR")
    if env_artifact_dir:
        artifact_dir = Path(env_artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_file, artifact_dir / "per_image_evaluation_results.csv")
        print(f"Copied CSV to artifact directory: {artifact_dir / 'per_image_evaluation_results.csv'}")

    return out_file


if __name__ == "__main__":
    generate_evaluation_csv()
