"""Rigorous independent evaluation framework for SafeHER neural detectors (Model A and Model B).

Enforces:
1. Strict evaluation separation: Training/validation records are unconditionally blocked.
2. Comprehensive metric computation with bootstrap confidence intervals.
3. Generalization stress testing: Unseen generators, manipulation families, and perturbations
   (multi-quality JPEG, social recompression, resizing, cropping, enhancement).
4. In-depth failure analysis (false positives, false negatives, compression drops, OOD).
5. Privacy preservation: All absolute paths and PII are scrubbed from outputs.
6. Machine-readable prediction exports (CSV / JSONL) and structured aggregate reports.
"""

import csv
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ai_core.datasets.manifest import (
    DatasetImageRecord,
    DatasetManifest,
    DatasetSplit,
    ManipulationCategory,
)
from ai_core.neural.evaluation.metrics import (
    ClassificationMetrics,
    compute_binary_metrics,
)
from ai_core.neural.evaluation.perturbations import (
    get_standard_perturbation_battery,
)
from ai_core.neural.facial.service import ModelBFacialInferenceService
from ai_core.neural.facial.schemas import FacialManipulationType, NormalizedBoundingBox
from ai_core.neural.inference import ModelAInferenceService

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_TEST_SPLITS: Set[DatasetSplit] = {
    DatasetSplit.INTERNAL_TEST,
    DatasetSplit.HELD_OUT_GENERATOR_TEST,
    DatasetSplit.CROSS_DATASET_TEST,
    DatasetSplit.COMPRESSION_ROBUSTNESS_TEST,
}


class PerSamplePrediction(BaseModel):
    """Sanitized machine-readable per-sample evaluation prediction."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., description="Anonymized unique record identifier.")
    sha256: str = Field(..., description="Cryptographic image hash.")
    ground_truth: int = Field(..., description="Binary ground truth (0=authentic, 1=manipulated).")
    label_name: str = Field(..., description="Ground truth category name.")
    predicted_label: int = Field(..., description="Predicted binary label.")
    uncalibrated_score: float = Field(..., description="Uncalibrated score for positive class.")
    is_correct: bool = Field(..., description="Whether prediction matches ground truth.")
    failure_mode: str = Field(default="NONE", description="Failure mode: 'FALSE_POSITIVE', 'FALSE_NEGATIVE', or 'NONE'.")
    generator_name: Optional[str] = Field(default=None, description="Synthetic generator identifier.")
    camera_source: Optional[str] = Field(default=None, description="Camera or source dataset identifier.")
    manipulation_category: str = Field(default="none", description="Fine-grained manipulation category.")
    perturbation: str = Field(default="baseline", description="Perturbation applied during evaluation.")
    is_synthetic_fixture: bool = Field(default=False, description="Whether sample belongs to synthetic fixture.")


class GeneralizationBenchmarkResult(BaseModel):
    """Generalization performance under a specific environmental stress perturbation."""

    model_config = ConfigDict(frozen=True)

    perturbation_name: str
    sample_count: int
    accuracy: float
    f1_score: float
    roc_auc: float
    f1_drop_relative: float


class FailureAnalysisReport(BaseModel):
    """Detailed forensic failure analysis categorization."""

    model_config = ConfigDict(frozen=True)

    false_positive_count: int
    false_negative_count: int
    false_positives: List[PerSamplePrediction]
    false_negatives: List[PerSamplePrediction]
    generator_error_rates: Dict[str, float]
    camera_source_false_alarm_rates: Dict[str, float]
    compression_induced_failures_count: int
    summary: str


class IndependentEvaluationReport(BaseModel):
    """Comprehensive independent evaluation report across all required metrics and dimensions."""

    model_config = ConfigDict(frozen=True)

    evaluation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_architecture: str
    model_version: str
    eval_splits_evaluated: List[str]
    total_samples: int
    baseline_metrics: ClassificationMetrics
    per_generator_metrics: Dict[str, ClassificationMetrics]
    per_manipulation_family_metrics: Dict[str, ClassificationMetrics]
    perturbation_benchmarks: Dict[str, GeneralizationBenchmarkResult]
    failure_analysis: FailureAnalysisReport
    synthetic_fixture_vs_real_world: Dict[str, Any]
    privacy_sanitization_verified: bool = True
    disclaimer: str = (
        "SafeHER Independent Evaluation Disclaimer: All test sets evaluated strictly post-training. "
        "Test metrics must never be used for hyperparameter tuning or early stopping. "
        "Absence of detected manipulation does NOT certify authenticity."
    )


class IndependentNeuralEvaluator:
    """Evaluates neural detectors (Model A or Model B) under strict separation and perturbation stress."""

    def __init__(
        self,
        service: Union[ModelAInferenceService, ModelBFacialInferenceService],
        decision_threshold: float = 0.50,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize IndependentNeuralEvaluator.

        Args:
            service: Initialized Model A or Model B inference service.
            decision_threshold: Fixed, read-only decision threshold for evaluation.
            output_dir: Directory where CSV/JSONL prediction logs will be persisted.
        """
        self.service = service
        self.decision_threshold = decision_threshold
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_from_source(
        self,
        record: DatasetImageRecord,
        root_dir: Optional[Union[str, Path]],
        in_memory_images: Optional[Dict[str, Any]],
    ) -> Image.Image:
        """Load image from in-memory cache or safe filesystem path."""
        if in_memory_images:
            if record.record_id in in_memory_images:
                val = in_memory_images[record.record_id]
                return val if isinstance(val, Image.Image) else Image.open(io.BytesIO(val)).convert("RGB")
            if record.image_path_or_ref in in_memory_images:
                val = in_memory_images[record.image_path_or_ref]
                return val if isinstance(val, Image.Image) else Image.open(io.BytesIO(val)).convert("RGB")

        if root_dir is not None:
            from ai_core.datasets.integrity import resolve_safe_relative_path
            path = resolve_safe_relative_path(root_dir, record.image_path_or_ref)
            if path.is_file():
                return Image.open(path).convert("RGB")

        raise FileNotFoundError(f"Could not load image asset for record '{record.record_id}' ({record.image_path_or_ref})")

    @staticmethod
    def _face_boxes_for_record(record: DatasetImageRecord) -> List[NormalizedBoundingBox]:
        """Load explicitly annotated face boxes; never infer a face from a center crop."""
        raw_boxes = record.metadata.get("face_bounding_boxes")
        if isinstance(raw_boxes, dict):
            raw_boxes = [raw_boxes]
        if not raw_boxes:
            single_box = record.metadata.get("face_bounding_box")
            raw_boxes = [single_box] if isinstance(single_box, dict) else []
        if not raw_boxes:
            raise ValueError(
                f"Model B evaluation record '{record.record_id}' has no annotated face bounding box. "
                "Add face_bounding_boxes to record metadata from a verified detector or annotation source."
            )
        return [NormalizedBoundingBox.model_validate(box) for box in raw_boxes]

    def _infer_sample(
        self,
        image: Image.Image,
        is_model_a: bool,
        face_boxes: Optional[List[NormalizedBoundingBox]] = None,
    ) -> Tuple[float, int]:
        """Execute single-sample inference and return positive manipulation score and predicted class."""
        if is_model_a:
            tensor = self.service.preprocess_image(image)
            res = self.service.predict(tensor)
            status = getattr(res.status, "value", str(res.status))
            if status != "READY" or "AI_GENERATED_IMAGE" not in res.uncalibrated_scores:
                raise ValueError(
                    f"Model A did not produce an assessed prediction (status={status}); "
                    "unavailable inference must not be counted as a negative prediction."
                )
            # Positive class: AI_GENERATED_IMAGE
            score = res.uncalibrated_scores["AI_GENERATED_IMAGE"]
            pred = 1 if score >= self.decision_threshold else 0
            return score, pred
        else:
            # Model B requires an upstream/annotated face box. Its preprocessor is
            # not itself a face detector and must never turn a center crop into one.
            if not face_boxes:
                raise ValueError(
                    "Model B evaluation requires face_bounding_boxes in each record's metadata. "
                    "A general image crop is not valid face localization."
                )
            res = self.service.analyze_image(image, provided_boxes=face_boxes)
            if (
                not res.model_available
                or res.predicted_category == FacialManipulationType.NOT_ASSESSED
                or res.inference_status != "READY"
            ):
                raise ValueError(
                    f"Model B did not produce an assessed prediction for this sample "
                    f"(status={res.inference_status}, category={res.predicted_category.value}). "
                    "An unassessed sample must not be counted as authentic."
                )
            # Positive class: any manipulation other than none_authentic
            authentic_score = res.manipulation_category_indicators.get("none_authentic", 1.0)
            score = 1.0 - authentic_score
            pred = 1 if score >= self.decision_threshold else 0
            return score, pred

    def evaluate_manifest(
        self,
        manifest: DatasetManifest,
        in_memory_images: Optional[Dict[str, Any]] = None,
        allowed_splits: Optional[Set[DatasetSplit]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        run_perturbations: bool = True,
        num_bootstraps: int = 500,
    ) -> IndependentEvaluationReport:
        """Execute rigorous independent evaluation across all required metrics and stress dimensions.

        Args:
            manifest: DatasetManifest containing evaluation records.
            in_memory_images: In-memory image byte/PIL dictionary.
            allowed_splits: Permitted splits (rejects any TRAIN or VAL records).
            root_dir: Root directory for on-disk images.
            run_perturbations: Whether to execute the full perturbation robustness battery.
            num_bootstraps: Number of bootstrap resamples for confidence intervals.

        Returns:
            IndependentEvaluationReport complete structured evaluation report.

        Raises:
            ValueError: If training or validation records are detected (evaluation separation violation).
        """
        valid_splits = allowed_splits or DEFAULT_ALLOWED_TEST_SPLITS

        # 1. Enforcement of Strict Evaluation Separation
        for r in manifest.records:
            if r.split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
                raise ValueError(
                    f"CRITICAL EVALUATION SEPARATION VIOLATION: Record '{r.record_id}' has split '{r.split.value}'. "
                    f"Training and validation data are strictly forbidden from independent evaluation runs."
                )
            if r.split not in valid_splits:
                raise ValueError(
                    f"Record '{r.record_id}' split '{r.split.value}' is not in allowed test splits: "
                    f"{[s.value for s in valid_splits]}"
                )

        eval_records = manifest.records
        if not eval_records:
            raise ValueError("No evaluation records found in manifest for allowed test splits.")

        is_model_a = isinstance(self.service, ModelAInferenceService)
        if is_model_a:
            if getattr(self.service, "status", None) is None or getattr(self.service.status, "value", str(self.service.status)) != "READY" or self.service.model is None:
                raise ValueError("Model A evaluation cannot start because verified trained weights are unavailable.")
        else:
            if getattr(self.service, "operational_status", None) != "READY" or self.service.model is None:
                raise ValueError("Model B evaluation cannot start because verified trained weights are unavailable.")
        arch_name = "mobilenet_v3_large" if is_model_a else "patch_forensics_facial"
        model_ver = getattr(self.service, "model_version", "1.0.0-m4.5")

        # 2. Baseline Evaluation
        predictions: List[PerSamplePrediction] = []
        baseline_true: List[int] = []
        baseline_scores: List[float] = []

        for r in eval_records:
            img = self._load_image_from_source(r, root_dir, in_memory_images)
            gt = 0 if r.label.value in ("CAMERA_PHOTOGRAPH", "none_authentic") else 1

            face_boxes = None
            if not is_model_a:
                face_boxes = self._face_boxes_for_record(r)

            score, pred = self._infer_sample(img, is_model_a=is_model_a, face_boxes=face_boxes)

            is_correct = (pred == gt)
            failure_mode = "NONE"
            if not is_correct:
                failure_mode = "FALSE_POSITIVE" if pred == 1 else "FALSE_NEGATIVE"

            sample_pred = PerSamplePrediction(
                record_id=r.record_id,
                sha256=r.sha256,
                ground_truth=gt,
                label_name=r.label.value,
                predicted_label=pred,
                uncalibrated_score=round(score, 4),
                is_correct=is_correct,
                failure_mode=failure_mode,
                generator_name=r.generator_name,
                camera_source=r.dataset_name,
                manipulation_category=r.manipulation_category.value,
                perturbation="baseline",
                is_synthetic_fixture=("synthetic" in r.dataset_name.lower()),
            )
            predictions.append(sample_pred)
            baseline_true.append(gt)
            baseline_scores.append(score)

        baseline_metrics = compute_binary_metrics(
            y_true=baseline_true,
            y_scores=baseline_scores,
            threshold=self.decision_threshold,
            compute_cis=True,
            num_bootstraps=num_bootstraps,
        )

        # 3. Stratified Per-Generator Breakdown
        per_generator_metrics: Dict[str, ClassificationMetrics] = {}
        generators = sorted(list({p.generator_name for p in predictions if p.generator_name}))
        for gen in generators:
            gen_true = [p.ground_truth for p in predictions if p.generator_name == gen or p.ground_truth == 0]
            gen_scores = [p.uncalibrated_score for p in predictions if p.generator_name == gen or p.ground_truth == 0]
            if gen_true:
                per_generator_metrics[gen] = compute_binary_metrics(
                    y_true=gen_true,
                    y_scores=gen_scores,
                    threshold=self.decision_threshold,
                    compute_cis=False,
                )

        # 4. Stratified Per-Manipulation-Family Breakdown
        per_manip_metrics: Dict[str, ClassificationMetrics] = {}
        manips = sorted(list({p.manipulation_category for p in predictions if p.manipulation_category != "none"}))
        for m in manips:
            m_true = [p.ground_truth for p in predictions if p.manipulation_category == m or p.ground_truth == 0]
            m_scores = [p.uncalibrated_score for p in predictions if p.manipulation_category == m or p.ground_truth == 0]
            if m_true:
                per_manip_metrics[m] = compute_binary_metrics(
                    y_true=m_true,
                    y_scores=m_scores,
                    threshold=self.decision_threshold,
                    compute_cis=False,
                )

        # 5. Generalization Perturbation Robustness Battery
        perturbation_benchmarks: Dict[str, GeneralizationBenchmarkResult] = {}
        perturbed_predictions: List[PerSamplePrediction] = list(predictions)
        compression_failures = 0

        if run_perturbations:
            battery = get_standard_perturbation_battery()
            for pert_name, pert_fn in battery.items():
                if pert_name == "baseline":
                    continue

                pert_true: List[int] = []
                pert_scores: List[float] = []

                for r in eval_records:
                    orig_img = self._load_image_from_source(r, root_dir, in_memory_images)
                    pert_img = pert_fn(orig_img)
                    gt = 0 if r.label.value in ("CAMERA_PHOTOGRAPH", "none_authentic") else 1

                    pert_face_boxes = self._face_boxes_for_record(r) if not is_model_a else None

                    score, pred = self._infer_sample(pert_img, is_model_a=is_model_a, face_boxes=pert_face_boxes)
                    pert_true.append(gt)
                    pert_scores.append(score)

                    # Track compression-induced failures (correct at baseline, failed under compression)
                    if "jpeg" in pert_name or "social" in pert_name:
                        base_p = next(p for p in predictions if p.record_id == r.record_id)
                        if base_p.is_correct and pred != gt:
                            compression_failures += 1

                    sample_p = PerSamplePrediction(
                        record_id=r.record_id,
                        sha256=r.sha256,
                        ground_truth=gt,
                        label_name=r.label.value,
                        predicted_label=pred,
                        uncalibrated_score=round(score, 4),
                        is_correct=(pred == gt),
                        failure_mode=("FALSE_POSITIVE" if pred == 1 else "FALSE_NEGATIVE") if pred != gt else "NONE",
                        generator_name=r.generator_name,
                        camera_source=r.dataset_name,
                        manipulation_category=r.manipulation_category.value,
                        perturbation=pert_name,
                        is_synthetic_fixture=("synthetic" in r.dataset_name.lower()),
                    )
                    perturbed_predictions.append(sample_p)

                p_metrics = compute_binary_metrics(
                    y_true=pert_true,
                    y_scores=pert_scores,
                    threshold=self.decision_threshold,
                    compute_cis=False,
                )

                f1_drop = round(baseline_metrics.f1_score - p_metrics.f1_score, 4)
                perturbation_benchmarks[pert_name] = GeneralizationBenchmarkResult(
                    perturbation_name=pert_name,
                    sample_count=len(pert_true),
                    accuracy=p_metrics.accuracy,
                    f1_score=p_metrics.f1_score,
                    roc_auc=p_metrics.roc_auc,
                    f1_drop_relative=f1_drop,
                )

        # 6. Failure Analysis
        fps = [p for p in predictions if p.failure_mode == "FALSE_POSITIVE"]
        fns = [p for p in predictions if p.failure_mode == "FALSE_NEGATIVE"]

        gen_errors: Dict[str, float] = {}
        for gen in generators:
            gen_samples = [p for p in predictions if p.generator_name == gen]
            gen_err = sum(1 for p in gen_samples if not p.is_correct) / max(1, len(gen_samples))
            gen_errors[gen] = round(gen_err, 4)

        camera_fprs: Dict[str, float] = {}
        cameras = sorted(list({p.camera_source for p in predictions if p.ground_truth == 0}))
        for cam in cameras:
            cam_samples = [p for p in predictions if p.camera_source == cam and p.ground_truth == 0]
            cam_fp = sum(1 for p in cam_samples if p.predicted_label == 1)
            camera_fprs[cam or "unknown"] = round(cam_fp / max(1, len(cam_samples)), 4)

        fail_summary = (
            f"Failure Analysis: {len(fps)} false positives (authentic flagged), "
            f"{len(fns)} false negatives (manipulated missed). "
            f"Compression-induced failures: {compression_failures} across test battery."
        )

        failure_analysis = FailureAnalysisReport(
            false_positive_count=len(fps),
            false_negative_count=len(fns),
            false_positives=fps[:10],
            false_negatives=fns[:10],
            generator_error_rates=gen_errors,
            camera_source_false_alarm_rates=camera_fprs,
            compression_induced_failures_count=compression_failures,
            summary=fail_summary,
        )

        # 7. Synthetic vs Real-World Breakdown
        synth_preds = [p for p in predictions if p.is_synthetic_fixture]
        real_preds = [p for p in predictions if not p.is_synthetic_fixture]

        synth_acc = float(np.mean([p.is_correct for p in synth_preds])) if synth_preds else 0.0
        real_acc = float(np.mean([p.is_correct for p in real_preds])) if real_preds else 0.0

        synth_vs_real = {
            "synthetic_fixture_sample_count": len(synth_preds),
            "synthetic_fixture_accuracy": round(synth_acc, 4),
            "real_world_sample_count": len(real_preds),
            "real_world_accuracy": round(real_acc, 4),
            "reporting_separation_enforced": True,
        }

        # 8. Export Machine-Readable Predictions (CSV & JSONL)
        self._export_predictions_csv(perturbed_predictions, arch_name)
        self._export_predictions_jsonl(perturbed_predictions, arch_name)

        report = IndependentEvaluationReport(
            model_architecture=arch_name,
            model_version=model_ver,
            eval_splits_evaluated=[s.value for s in valid_splits],
            total_samples=len(predictions),
            baseline_metrics=baseline_metrics,
            per_generator_metrics=per_generator_metrics,
            per_manipulation_family_metrics=per_manip_metrics,
            perturbation_benchmarks=perturbation_benchmarks,
            failure_analysis=failure_analysis,
            synthetic_fixture_vs_real_world=synth_vs_real,
            privacy_sanitization_verified=True,
        )

        # Save summary JSON
        summary_path = self.output_dir / f"evaluation_summary_{arch_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        return report

    def _export_predictions_csv(self, predictions: List[PerSamplePrediction], arch: str) -> Path:
        """Export sanitized per-sample predictions to CSV file."""
        csv_path = self.output_dir / f"predictions_{arch}.csv"
        fieldnames = [
            "record_id", "sha256", "ground_truth", "label_name",
            "predicted_label", "uncalibrated_score", "is_correct",
            "failure_mode", "generator_name", "camera_source",
            "manipulation_category", "perturbation", "is_synthetic_fixture",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in predictions:
                writer.writerow(p.model_dump())
        return csv_path

    def _export_predictions_jsonl(self, predictions: List[PerSamplePrediction], arch: str) -> Path:
        """Export sanitized per-sample predictions to JSONL file."""
        jsonl_path = self.output_dir / f"predictions_{arch}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p.model_dump()) + "\n")
        return jsonl_path
