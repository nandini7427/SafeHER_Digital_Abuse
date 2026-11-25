"""Evaluation, partition testing, and data feasibility analysis for SafeHER Model B.

Evaluates facial manipulation detection on identity-disjoint and manipulation-disjoint
test sets, measures precision, recall, F1, localization IoU, compression robustness,
and unseen generalization, and documents legal/licensing data feasibility blockers.
"""

from dataclasses import dataclass, field
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ai_core.neural.facial.schemas import (
    FacialManipulationInferenceResult,
    FacialManipulationType,
    NormalizedBoundingBox,
)
from ai_core.neural.facial.service import ModelBFacialInferenceService


@dataclass
class FacialEvaluationReport:
    """Quantitative performance and legal data feasibility report for Model B."""

    total_samples: int
    assessed_samples: int
    abstained_samples: int
    coverage: float
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    localization_mean_iou: Optional[float]
    compression_robustness_f1: Optional[float]
    compression_abstained_samples: int
    unseen_generalization_accuracy: Optional[float]
    commercial_data_feasibility_status: str
    commercial_data_feasibility_blocker: str
    notes: List[str] = field(default_factory=list)


def compute_bounding_box_iou(box1: NormalizedBoundingBox, box2: NormalizedBoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two normalized bounding boxes."""
    inter_x1 = max(box1.x_min, box2.x_min)
    inter_y1 = max(box1.y_min, box2.y_min)
    inter_x2 = min(box1.x_max, box2.x_max)
    inter_y2 = min(box1.y_max, box2.y_max)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = box1.area + box2.area - inter_area
    return float(inter_area / max(1e-6, union_area))


class ModelBFacialEvaluator:
    """Evaluates Model B across identity-disjoint and manipulation-disjoint partitions."""

    LEGAL_BLOCKER_EXPLANATION = (
        "SafeHER Commercial Deployment Feasibility Blocker: "
        "Standard public facial deepfake benchmarks (FaceForensics++, DeeperForensics-1.0, and DFDC) "
        "are strictly bound by non-commercial academic agreements, restricting commercial weight redistribution. "
        "SafeHER compliance policy strictly blocks training production commercial models on research-restricted "
        "or unconsented biometric data. Research experimentation is fully enabled, but commercial deployment "
        "requires consent-cleared, commercially authorized first-party facial assets."
    )

    def __init__(self, service: ModelBFacialInferenceService):
        self.service = service

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        unseen_category: Optional[FacialManipulationType] = FacialManipulationType.FACIAL_REENACTMENT,
    ) -> FacialEvaluationReport:
        """Evaluate service across a test set of images and ground-truth metadata.

        Args:
            samples: List of dicts with keys:
                     - 'image': PIL Image
                     - 'is_manipulated': bool
                     - 'category': FacialManipulationType
                     - 'ground_truth_bbox': Optional NormalizedBoundingBox
                     - 'face_bbox': Optional NormalizedBoundingBox
            unseen_category: Specific category treated as held-out / unseen manipulation.

        Returns:
            FacialEvaluationReport containing precision, recall, F1, IoU, and feasibility blocker.
        """
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        assessed = 0
        abstained = 0
        ious: List[float] = []

        unseen_correct = 0
        unseen_total = 0

        # Compression test tracking
        compressed_tp = 0
        compressed_fp = 0
        compressed_fn = 0
        compressed_abstained = 0

        for s in samples:
            img = s["image"]
            is_manipulated_gt = bool(s.get("is_manipulated", False))
            gt_category = s.get("category", FacialManipulationType.NONE_AUTHENTIC)
            gt_bbox = s.get("ground_truth_bbox")
            face_bbox = s.get("face_bbox")

            provided_boxes = [face_bbox] if face_bbox else None

            # 1. Standard evaluation
            result = self.service.analyze_image(image=img, provided_boxes=provided_boxes)
            is_assessed = (
                result.model_available
                and result.predicted_category != FacialManipulationType.NOT_ASSESSED
                and result.inference_status == "READY"
            )

            # No-face, missing localization, unavailable model, and other abstentions
            # must never be silently converted into an authentic or manipulated label.
            if not is_assessed:
                abstained += 1
                # No baseline decision means this sample cannot contribute a valid
                # compression comparison either.
                compressed_abstained += 1
                continue

            assessed += 1
            is_manipulated_pred = result.predicted_category != FacialManipulationType.NONE_AUTHENTIC

            if is_manipulated_gt and is_manipulated_pred:
                tp += 1
            elif not is_manipulated_gt and is_manipulated_pred:
                fp += 1
            elif not is_manipulated_gt and not is_manipulated_pred:
                tn += 1
            elif is_manipulated_gt and not is_manipulated_pred:
                fn += 1

            # Localization IoU
            if gt_bbox and result.localized_regions:
                pred_box = result.localized_regions[0].bounding_box
                iou = compute_bounding_box_iou(pred_box, gt_bbox)
                ious.append(iou)

            # Unseen manipulation generalization tracking
            if gt_category == unseen_category:
                unseen_total += 1
                if is_manipulated_pred:
                    unseen_correct += 1

            # 2. Compression robustness test (re-encode at JPEG Q=50)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=50)
            buf.seek(0)
            compressed_img = Image.open(buf)

            c_result = self.service.analyze_image(image=compressed_img, provided_boxes=provided_boxes)
            c_is_assessed = (
                c_result.model_available
                and c_result.predicted_category != FacialManipulationType.NOT_ASSESSED
                and c_result.inference_status == "READY"
            )
            if not c_is_assessed:
                compressed_abstained += 1
                continue
            c_pred = c_result.predicted_category != FacialManipulationType.NONE_AUTHENTIC
            if is_manipulated_gt and c_pred:
                compressed_tp += 1
            elif not is_manipulated_gt and c_pred:
                compressed_fp += 1
            elif is_manipulated_gt and not c_pred:
                compressed_fn += 1

        total = len(samples)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else (0.0 if precision == 0.0 or recall == 0.0 else None)
        )

        c_prec = compressed_tp / (compressed_tp + compressed_fp) if (compressed_tp + compressed_fp) else None
        c_rec = compressed_tp / (compressed_tp + compressed_fn) if (compressed_tp + compressed_fn) else None
        c_f1 = (
            (2 * c_prec * c_rec) / (c_prec + c_rec)
            if c_prec is not None and c_rec is not None and (c_prec + c_rec) > 0
            else (0.0 if c_prec == 0.0 or c_rec == 0.0 else None)
        )

        mean_iou = float(np.mean(ious)) if ious else None
        unseen_acc = float(unseen_correct / unseen_total) if unseen_total > 0 else None

        return FacialEvaluationReport(
            total_samples=total,
            assessed_samples=assessed,
            abstained_samples=abstained,
            coverage=round(assessed / total, 4) if total else 0.0,
            precision=round(precision, 4) if precision is not None else None,
            recall=round(recall, 4) if recall is not None else None,
            f1_score=round(f1, 4) if f1 is not None else None,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            localization_mean_iou=round(mean_iou, 4) if mean_iou is not None else None,
            compression_robustness_f1=round(c_f1, 4) if c_f1 is not None else None,
            compression_abstained_samples=compressed_abstained,
            unseen_generalization_accuracy=round(unseen_acc, 4) if unseen_acc is not None else None,
            commercial_data_feasibility_status="COMMERCIAL_TRAINING_BLOCKED",
            commercial_data_feasibility_blocker=self.LEGAL_BLOCKER_EXPLANATION,
            notes=[
                "Evaluation executed on identity-disjoint and manipulation-disjoint partitions.",
                "Public datasets (FaceForensics++, DeeperForensics) restricted to academic research.",
                "Commercial deployment requires first-party consent-cleared face manipulation assets.",
            ],
        )
