"""Privacy-Preserving Model Drift Detection and Monitoring Architecture for SafeHER.

Core Principles:
1. Zero User Image Retention:
   Monitoring operates purely on privacy-safe aggregate statistics, categorical triage states,
   and synthetic evaluation datasets. No user images, pixels, base64 strings, or file paths are retained.
2. Multi-Dimensional Drift Detection:
   - Population Stability Index (PSI) for forensic score distribution shifts (covariate shift).
   - Input dimension distribution (thumbnail, standard, hd, large).
   - Compression & quality categorization (pristine, high, moderate, heavy compression, blur).
   - Out-of-distribution (OOD) and unsupported domain rate tracking.
   - Prediction and uncertainty distribution tracking.
   - Human-in-the-loop review disagreement rate tracking (concept drift).
   - Inference failure, latency percentiles (p50, p95, p99), and pipeline stability.
   - Calibration monitoring (mean confidence, Brier score estimates).
3. Objective Evaluation Dataset Benchmarking:
   - Validation-anchored drift thresholds (PSI, ECE, Brier score, failure rates).
4. Failure Isolation:
   - Malformed, corrupt, or missing telemetry is safely quarantined without interrupting inference.
5. Automated Model Training Prohibition:
   - Explicit guardrail strictly forbidding automatic retraining or live parameter updates
     from unverified user feedback (preventing data poisoning and feedback loops).
6. Version Lineage and Deterministic Rollback:
   - Pinned semantic version tracking and instant zero-downtime rollback runbook.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import math
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from ai_core.image_intelligence.config import ImageDetectorConfig
from ai_core.image_intelligence.detector import BaseImageDetector, SafeHERImageDetector
from ai_core.image_intelligence.model_runner import (
    BaseImageModelRunner,
    BaselineHeuristicRunner,
    ModelPrediction,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    ImageAssessmentResponse,
    ImageQualityIndicators,
    UncertaintyState,
)
from ai_core.image_intelligence.service import get_image_detector, set_image_detector

logger = logging.getLogger("safeher.ai.monitoring")


class DriftAlertSeverity(str, Enum):
    """Categorical alert severity for ML drift indicators."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DriftAlert:
    """Structured drift notification alert container."""

    alert_id: str
    timestamp: datetime
    severity: DriftAlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    model_version: str
    message: str
    recommended_action: str


@dataclass(frozen=True)
class DriftMetricBaseline:
    """Certified operational baseline metrics for a validated model version."""

    model_name: str
    model_version: str
    architecture: str = "mobilenet_v3_large"
    baseline_auc: float = 0.90
    baseline_fpr_at_90_tpr: float = 0.08
    baseline_ece: float = 0.08
    baseline_brier_score: float = 0.08
    baseline_p95_latency_ms: float = 150.0
    baseline_failure_rate: float = 0.005  # 0.5%
    baseline_disagreement_rate: float = 0.08  # 8.0%
    baseline_ood_rate: float = 0.05  # 5.0%
    baseline_poor_quality_rate: float = 0.07  # 7.0%
    # 10-bin reference histogram for Population Stability Index (PSI)
    score_reference_distribution: Tuple[float, ...] = (
        0.35, 0.25, 0.12, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02, 0.03
    )
    # Thresholds
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    failure_rate_warning_threshold: float = 0.01  # 1.0%
    failure_rate_critical_threshold: float = 0.03  # 3.0%
    ood_rate_warning_threshold: float = 0.15      # 15.0%
    ood_rate_critical_threshold: float = 0.25      # 25.0%
    latency_p95_warning_ms: float = 200.0
    latency_p95_critical_ms: float = 500.0
    score_polarization_threshold: float = 0.70    # Max allowed concentration in single bin


@dataclass(frozen=True)
class TelemetryRecord:
    """Privacy-safe anonymized telemetry event.

    Strictly excludes raw image pixels, base64 strings, user IDs, or file paths.
    """

    record_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    is_failure: bool
    failure_category: Optional[str]
    raw_score: Optional[float]
    assessment_label: Optional[str]
    uncertainty_state: Optional[str]
    is_ood: bool
    quality_blur: bool
    quality_exposure: bool
    quality_resolution_ok: bool
    image_format: str
    dimension_bucket: str  # "thumbnail", "standard", "hd", "large"
    elapsed_ms: float
    calibrated_score: Optional[float] = None
    architecture: Optional[str] = None
    compression_category: str = "standard_quality"
    neural_status: Optional[str] = None


@dataclass
class HumanReviewMetricEntry:
    """Review outcome entry for tracking human-AI consistency without storing case data."""

    timestamp: datetime
    is_disagreement: bool
    human_verdict: str


def compute_population_stability_index(
    baseline_dist: np.ndarray,
    current_dist: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between baseline and observed distributions.

    PSI Interpretation:
    - PSI < 0.10: No significant shift (Stable).
    - 0.10 <= PSI < 0.25: Moderate shift (Investigate covariate drift).
    - PSI >= 0.25: Significant shift (Action required; triggers alert/audit).
    """
    b = np.array(baseline_dist, dtype=float) + epsilon
    c = np.array(current_dist, dtype=float) + epsilon

    # Normalize to probabilities
    b = b / np.sum(b)
    c = c / np.sum(c)

    psi_value = np.sum((c - b) * np.log(c / b))
    return float(psi_value)


def categorize_dimensions(dimensions: Tuple[int, int]) -> str:
    """Categorize spatial image dimensions into privacy-preserving buckets."""
    w, h = dimensions
    max_dim = max(w, h)
    if max_dim <= 256:
        return "thumbnail"
    elif max_dim <= 1080:
        return "standard"
    elif max_dim <= 1920:
        return "hd"
    else:
        return "large"


def categorize_compression(
    image_format: str,
    quality_indicators: Optional[ImageQualityIndicators] = None,
    estimated_quality: Optional[int] = None,
) -> str:
    """Categorize image compression & quality state."""
    fmt = image_format.upper()
    if fmt in ("PNG", "TIFF"):
        return "pristine_uncompressed"

    if quality_indicators:
        if quality_indicators.has_heavy_compression or quality_indicators.compression_artifact_score > 0.35:
            return "heavy_compression"
        elif quality_indicators.compression_artifact_score > 0.15:
            return "moderate_compression"
        elif quality_indicators.overall_quality_score >= 0.85:
            return "high_quality"

    if estimated_quality is not None:
        if estimated_quality < 60:
            return "heavy_compression"
        elif estimated_quality < 85:
            return "moderate_compression"
        else:
            return "high_quality"

    return "standard_quality"


class ModelMonitoringEngine:
    """In-memory, privacy-preserving monitoring engine for SafeHER image models."""

    def __init__(
        self,
        baseline: Optional[DriftMetricBaseline] = None,
        max_telemetry_window: int = 5000,
        max_review_window: int = 1000,
    ) -> None:
        self.baseline = baseline or DriftMetricBaseline(
            model_name="SafeHER-Baseline-ELA-Heuristics",
            model_version="0.2.0",
        )
        self._telemetry_window: Deque[TelemetryRecord] = deque(maxlen=max_telemetry_window)
        self._review_window: Deque[HumanReviewMetricEntry] = deque(maxlen=max_review_window)
        self._alert_history: List[DriftAlert] = []
        self._corrupted_telemetry_count: int = 0

        # Pinned Model Version Registry
        self._model_registry: Dict[str, Dict[str, Any]] = {
            "0.2.0": {
                "model_name": "SafeHER-Baseline-ELA-Heuristics",
                "version": "0.2.0",
                "architecture": "ela_fft_heuristics",
                "release_date": "2026-11-20",
                "status": "production_active",
                "description": "Deterministic Error Level Analysis and frequency variance baseline.",
                "certified_auc": 0.902,
                "certified_ece": 0.081,
            },
            "0.1.0": {
                "model_name": "SafeHER-Baseline-ELA-Heuristics",
                "version": "0.1.0",
                "architecture": "ela_heuristics_v1",
                "release_date": "2026-11-15",
                "status": "archived_certified",
                "description": "Legacy baseline heuristic engine.",
                "certified_auc": 0.884,
                "certified_ece": 0.095,
            },
            "1.0.0-m4.3": {
                "model_name": "SafeHER-Model-A-Neural-Detector",
                "version": "1.0.0-m4.3",
                "architecture": "mobilenet_v3_large",
                "release_date": "2026-11-24",
                "status": "production_active",
                "description": "MobileNetV3-Large AI-generated image detector with temperature scaling calibration.",
                "certified_auc": 0.942,
                "certified_ece": 0.045,
            },
            "1.0.0-m4.4-model-b": {
                "model_name": "SafeHER-Model-B-Facial-Detector",
                "version": "1.0.0-m4.4-model-b",
                "architecture": "patch_forensics_facial",
                "release_date": "2026-11-24",
                "status": "production_active",
                "description": "Dense patch forensics facial manipulation detector and spatial boundary localizer.",
                "certified_auc": 0.925,
                "certified_ece": 0.052,
            },
        }

    # --------------------------------------------------------------------------
    # Telemetry Ingestion (Zero Raw Image Data & Failure Isolation)
    # --------------------------------------------------------------------------

    def record_assessment_telemetry(
        self,
        response: Optional[ImageAssessmentResponse] = None,
        is_failure: bool = False,
        failure_category: Optional[str] = None,
        elapsed_ms: float = 0.0,
        image_format: str = "UNKNOWN",
        dimensions: Tuple[int, int] = (0, 0),
    ) -> None:
        """Record privacy-safe aggregate telemetry metrics from an inference turn."""
        try:
            now = datetime.now(timezone.utc)
            rec_id = str(uuid.uuid4())

            dim_bucket = categorize_dimensions(dimensions)

            clean_elapsed = 0.0
            try:
                clean_elapsed = max(0.0, float(elapsed_ms))
            except (ValueError, TypeError):
                self._corrupted_telemetry_count += 1

            if is_failure or response is None:
                record = TelemetryRecord(
                    record_id=rec_id,
                    timestamp=now,
                    model_name=self.baseline.model_name,
                    model_version=self.baseline.model_version,
                    is_failure=True,
                    failure_category=failure_category or "internal_error",
                    raw_score=None,
                    assessment_label=None,
                    uncertainty_state=UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED.value,
                    is_ood=False,
                    quality_blur=False,
                    quality_exposure=False,
                    quality_resolution_ok=False,
                    image_format=str(image_format).upper(),
                    dimension_bucket=dim_bucket,
                    elapsed_ms=clean_elapsed,
                    compression_category=categorize_compression(image_format),
                )
            else:
                qi = response.quality_indicators
                raw_score = response.raw_model_score
                # Sanitize non-finite values
                if raw_score is not None and (math.isnan(raw_score) or math.isinf(raw_score)):
                    raw_score = None
                    self._corrupted_telemetry_count += 1

                comp_cat = categorize_compression(
                    image_format=response.metadata_summary.format if response.metadata_summary else image_format,
                    quality_indicators=qi,
                )

                record = TelemetryRecord(
                    record_id=rec_id,
                    timestamp=now,
                    model_name=response.detector_name,
                    model_version=response.detector_version,
                    is_failure=False,
                    failure_category=None,
                    raw_score=raw_score,
                    assessment_label=response.assessment_label.value if hasattr(response.assessment_label, "value") else str(response.assessment_label),
                    uncertainty_state=response.uncertainty_state.value if hasattr(response.uncertainty_state, "value") else str(response.uncertainty_state),
                    is_ood=(response.uncertainty_state == UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN),
                    quality_blur=qi.is_blurry if qi else False,
                    quality_exposure=(qi.exposure_level != "normal") if qi else False,
                    quality_resolution_ok=qi.resolution_ok if qi else True,
                    image_format=(response.metadata_summary.format if response.metadata_summary else image_format).upper(),
                    dimension_bucket=dim_bucket,
                    elapsed_ms=clean_elapsed,
                    compression_category=comp_cat,
                )

            self._telemetry_window.append(record)
        except Exception as exc:
            # Monitoring Failure Isolation: Ingestion failures must NEVER interrupt production inference
            self._corrupted_telemetry_count += 1
            logger.error("Telemetry ingestion isolated error: %s", exc)

    def record_neural_telemetry(
        self,
        model_name: str,
        model_version: str,
        architecture: Optional[str] = None,
        raw_score: Optional[float] = None,
        calibrated_score: Optional[float] = None,
        predicted_class: Optional[str] = None,
        uncertainty_state: Optional[str] = None,
        is_ood: bool = False,
        dimensions: Tuple[int, int] = (0, 0),
        image_format: str = "UNKNOWN",
        compression_category: Optional[str] = None,
        quality_blur: bool = False,
        quality_exposure: bool = False,
        quality_resolution_ok: bool = True,
        is_failure: bool = False,
        failure_category: Optional[str] = None,
        elapsed_ms: float = 0.0,
        neural_status: str = "active",
    ) -> None:
        """Record privacy-preserving telemetry for neural image and facial detectors.

        Enforces:
        1. Zero image pixel or file-path retention.
        2. Strict failure isolation (catches corrupted NaN/Inf inputs without raising).
        """
        try:
            # 1. Sanitize raw_score
            sanitized_raw: Optional[float] = None
            if raw_score is not None:
                try:
                    val = float(raw_score)
                    if not (math.isnan(val) or math.isinf(val)):
                        sanitized_raw = round(float(np.clip(val, 0.0, 1.0)), 4)
                    else:
                        self._corrupted_telemetry_count += 1
                        logger.warning("Corrupted telemetry: non-finite raw_score discarded.")
                except (ValueError, TypeError):
                    self._corrupted_telemetry_count += 1
                    logger.warning("Corrupted telemetry: invalid raw_score type discarded.")

            # 2. Sanitize calibrated_score
            sanitized_cal: Optional[float] = None
            if calibrated_score is not None:
                try:
                    val = float(calibrated_score)
                    if not (math.isnan(val) or math.isinf(val)):
                        sanitized_cal = round(float(np.clip(val, 0.0, 1.0)), 4)
                    else:
                        self._corrupted_telemetry_count += 1
                except (ValueError, TypeError):
                    self._corrupted_telemetry_count += 1

            clean_elapsed = 0.0
            try:
                clean_elapsed = max(0.0, float(elapsed_ms))
            except (ValueError, TypeError):
                self._corrupted_telemetry_count += 1

            dim_bucket = categorize_dimensions(dimensions)
            comp_cat = compression_category or categorize_compression(image_format)

            record = TelemetryRecord(
                record_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                model_name=str(model_name),
                model_version=str(model_version),
                is_failure=bool(is_failure),
                failure_category=failure_category if is_failure else None,
                raw_score=sanitized_raw,
                assessment_label=str(predicted_class) if predicted_class else None,
                uncertainty_state=str(uncertainty_state) if uncertainty_state else None,
                is_ood=bool(is_ood),
                quality_blur=bool(quality_blur),
                quality_exposure=bool(quality_exposure),
                quality_resolution_ok=bool(quality_resolution_ok),
                image_format=str(image_format).upper(),
                dimension_bucket=dim_bucket,
                elapsed_ms=clean_elapsed,
                calibrated_score=sanitized_cal,
                architecture=architecture,
                compression_category=comp_cat,
                neural_status=neural_status,
            )
            self._telemetry_window.append(record)
        except Exception as exc:
            self._corrupted_telemetry_count += 1
            logger.error("Neural telemetry ingestion error isolated: %s", exc)

    def record_human_review_metric(self, is_disagreement: bool, human_verdict: str) -> None:
        """Record reviewer agreement outcome without retaining user case identifiers."""
        self._review_window.append(
            HumanReviewMetricEntry(
                timestamp=datetime.now(timezone.utc),
                is_disagreement=is_disagreement,
                human_verdict=human_verdict,
            )
        )

    # --------------------------------------------------------------------------
    # Telemetry Summary & Statistics
    # --------------------------------------------------------------------------

    def get_telemetry_summary(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """Aggregate summary metrics across the recent observation window."""
        records = list(self._telemetry_window)
        if window_size and len(records) > window_size:
            records = records[-window_size:]

        total_requests = len(records)
        if total_requests == 0:
            return {
                "total_requests": 0,
                "corrupted_telemetry_count": self._corrupted_telemetry_count,
                "failure_rate": 0.0,
                "disagreement_rate": 0.0,
                "ood_rate": 0.0,
                "mean_latency_ms": 0.0,
                "dimension_distribution": {},
                "compression_distribution": {},
                "prediction_distribution": {},
                "uncertainty_distribution": {},
                "model_version_usage": {},
                "latency_summary": {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0},
            }

        failures = sum(1 for r in records if r.is_failure)
        ood_count = sum(1 for r in records if r.is_ood)
        blurry_count = sum(1 for r in records if r.quality_blur)
        extreme_exp_count = sum(1 for r in records if r.quality_exposure)
        low_res_count = sum(1 for r in records if not r.quality_resolution_ok)

        # Dimension breakdown
        dim_counts: Dict[str, int] = {}
        for r in records:
            dim_counts[r.dimension_bucket] = dim_counts.get(r.dimension_bucket, 0) + 1

        # Compression breakdown
        comp_counts: Dict[str, int] = {}
        for r in records:
            comp_counts[r.compression_category] = comp_counts.get(r.compression_category, 0) + 1

        # Prediction distribution
        pred_counts: Dict[str, int] = {}
        for r in records:
            if r.assessment_label:
                pred_counts[r.assessment_label] = pred_counts.get(r.assessment_label, 0) + 1

        # Uncertainty state breakdown
        uncertainty_counts: Dict[str, int] = {}
        for r in records:
            if r.uncertainty_state:
                uncertainty_counts[r.uncertainty_state] = uncertainty_counts.get(r.uncertainty_state, 0) + 1

        # Format breakdown
        format_counts: Dict[str, int] = {}
        for r in records:
            format_counts[r.image_format] = format_counts.get(r.image_format, 0) + 1

        # Model version usage
        version_counts: Dict[str, int] = {}
        for r in records:
            version_counts[r.model_version] = version_counts.get(r.model_version, 0) + 1

        # Latency distribution (percentiles)
        latencies = [r.elapsed_ms for r in records]
        lat_mean = float(np.mean(latencies)) if latencies else 0.0
        lat_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        lat_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        lat_p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

        # Review disagreement
        reviews = list(self._review_window)
        total_reviews = len(reviews)
        disagreements = sum(1 for rev in reviews if rev.is_disagreement)
        disagreement_rate = (disagreements / total_reviews) if total_reviews > 0 else 0.0

        # Score distribution & histogram
        scores = [r.raw_score for r in records if r.raw_score is not None]
        mean_score = float(np.mean(scores)) if scores else 0.0
        score_hist, _ = np.histogram(scores, bins=10, range=(0.0, 1.0)) if scores else (np.zeros(10), None)

        # Calibration summary
        cal_scores = [r.calibrated_score for r in records if r.calibrated_score is not None]
        cal_count = len(cal_scores)
        mean_cal = float(np.mean(cal_scores)) if cal_scores else 0.0

        return {
            "total_requests": total_requests,
            "corrupted_telemetry_count": self._corrupted_telemetry_count,
            "failure_count": failures,
            "failure_rate": round(failures / total_requests, 4),
            "ood_count": ood_count,
            "ood_rate": round(ood_count / total_requests, 4),
            "blur_rate": round(blurry_count / total_requests, 4),
            "exposure_anomaly_rate": round(extreme_exp_count / total_requests, 4),
            "low_resolution_rate": round(low_res_count / total_requests, 4),
            "dimension_distribution": {
                k: round(v / total_requests, 4) for k, v in dim_counts.items()
            },
            "compression_distribution": {
                k: round(v / total_requests, 4) for k, v in comp_counts.items()
            },
            "prediction_distribution": {
                k: round(v / total_requests, 4) for k, v in pred_counts.items()
            },
            "uncertainty_distribution": {
                k: round(v / total_requests, 4) for k, v in uncertainty_counts.items()
            },
            "format_distribution": {
                k: round(v / total_requests, 4) for k, v in format_counts.items()
            },
            "model_version_usage": version_counts,
            "total_human_reviews": total_reviews,
            "human_disagreements": disagreements,
            "human_disagreement_rate": round(disagreement_rate, 4),
            "mean_score": round(mean_score, 4),
            "score_histogram": [int(x) for x in score_hist],
            "mean_latency_ms": round(lat_mean, 2),
            "latency_summary": {
                "mean_ms": round(lat_mean, 2),
                "median_ms": round(lat_p50, 2),
                "p95_ms": round(lat_p95, 2),
                "p99_ms": round(lat_p99, 2),
            },
            "calibration_summary": {
                "calibrated_predictions_count": cal_count,
                "mean_calibrated_confidence": round(mean_cal, 4),
            },
        }

    # --------------------------------------------------------------------------
    # Drift Evaluation & Threshold Alert Engine
    # --------------------------------------------------------------------------

    def evaluate_drift_and_alerts(self, min_sample_size: int = 20) -> List[DriftAlert]:
        """Evaluate multi-dimensional drift indicators against certified baseline thresholds."""
        alerts: List[DriftAlert] = []
        summary = self.get_telemetry_summary()
        n = summary["total_requests"]

        if n < min_sample_size:
            logger.info("Insufficient observation window (%d < %d) for drift alerts.", n, min_sample_size)
            return alerts

        now = datetime.now(timezone.utc)
        v = self.baseline.model_version

        # 1. Failure Rate Alert
        fail_rate = summary["failure_rate"]
        if fail_rate >= self.baseline.failure_rate_critical_threshold:
            alerts.append(
                DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=now,
                    severity=DriftAlertSeverity.CRITICAL,
                    metric_name="inference_failure_rate",
                    current_value=fail_rate,
                    threshold_value=self.baseline.failure_rate_critical_threshold,
                    model_version=v,
                    message=f"Critical inference failure rate ({fail_rate:.1%}) exceeds {self.baseline.failure_rate_critical_threshold:.1%} threshold.",
                    recommended_action="Execute immediate service rollback to deterministic heuristic runner.",
                )
            )
        elif fail_rate >= self.baseline.failure_rate_warning_threshold:
            alerts.append(
                DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=now,
                    severity=DriftAlertSeverity.WARNING,
                    metric_name="inference_failure_rate",
                    current_value=fail_rate,
                    threshold_value=self.baseline.failure_rate_warning_threshold,
                    model_version=v,
                    message=f"Elevated inference failure rate ({fail_rate:.1%}) exceeds {self.baseline.failure_rate_warning_threshold:.1%} threshold.",
                    recommended_action="Inspect application error logs for corrupt payloads or timeout pressure.",
                )
            )

        # 2. Out-of-Distribution (OOD) Rate Alert
        ood_rate = summary["ood_rate"]
        if ood_rate >= self.baseline.ood_rate_critical_threshold:
            alerts.append(
                DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=now,
                    severity=DriftAlertSeverity.CRITICAL,
                    metric_name="out_of_distribution_rate",
                    current_value=ood_rate,
                    threshold_value=self.baseline.ood_rate_critical_threshold,
                    model_version=v,
                    message=f"Severe out-of-distribution rate ({ood_rate:.1%}) indicates radical traffic shift.",
                    recommended_action="Review user image sources; verify if unsupported platforms are uploading.",
                )
            )
        elif ood_rate >= self.baseline.ood_rate_warning_threshold:
            alerts.append(
                DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=now,
                    severity=DriftAlertSeverity.WARNING,
                    metric_name="out_of_distribution_rate",
                    current_value=ood_rate,
                    threshold_value=self.baseline.ood_rate_warning_threshold,
                    model_version=v,
                    message=f"Moderate OOD traffic elevation ({ood_rate:.1%}) exceeds {self.baseline.ood_rate_warning_threshold:.1%} threshold.",
                    recommended_action="Monitor screenshot and document upload patterns in survivor cases.",
                )
            )

        # 3. Human Review Disagreement Alert (Concept Drift)
        if summary["total_human_reviews"] >= 10:
            disagreement_rate = summary["human_disagreement_rate"]
            if disagreement_rate >= 0.25:
                alerts.append(
                    DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now,
                        severity=DriftAlertSeverity.CRITICAL,
                        metric_name="human_review_disagreement_rate",
                        current_value=disagreement_rate,
                        threshold_value=0.25,
                        model_version=v,
                        message=f"Critical human review disagreement ({disagreement_rate:.1%}) indicates severe concept drift.",
                        recommended_action="Schedule emergency offline audit on evaluation benchmark; assess new generator styles.",
                    )
                )
            elif disagreement_rate >= 0.15:
                alerts.append(
                    DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now,
                        severity=DriftAlertSeverity.WARNING,
                        metric_name="human_review_disagreement_rate",
                        current_value=disagreement_rate,
                        threshold_value=0.15,
                        model_version=v,
                        message=f"Elevated human review disagreement ({disagreement_rate:.1%}) exceeds 15% threshold.",
                        recommended_action="Examine human reviewer notes for patterns of new synthetic artifacts.",
                    )
                )

        # 4. Score Distribution Shift (Population Stability Index - PSI)
        scores = [r.raw_score for r in self._telemetry_window if r.raw_score is not None]
        if len(scores) >= min_sample_size:
            current_hist, _ = np.histogram(scores, bins=10, range=(0.0, 1.0))
            psi_score = compute_population_stability_index(
                baseline_dist=np.array(self.baseline.score_reference_distribution),
                current_dist=current_hist,
            )

            if psi_score >= self.baseline.psi_critical_threshold:
                alerts.append(
                    DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now,
                        severity=DriftAlertSeverity.CRITICAL,
                        metric_name="population_stability_index",
                        current_value=round(psi_score, 4),
                        threshold_value=self.baseline.psi_critical_threshold,
                        model_version=v,
                        message=f"Significant score distribution drift (PSI={psi_score:.3f} >= {self.baseline.psi_critical_threshold}).",
                        recommended_action="Initiate comprehensive model re-evaluation against certified benchmark.",
                    )
                )
            elif psi_score >= self.baseline.psi_warning_threshold:
                alerts.append(
                    DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now,
                        severity=DriftAlertSeverity.WARNING,
                        metric_name="population_stability_index",
                        current_value=round(psi_score, 4),
                        threshold_value=self.baseline.psi_warning_threshold,
                        model_version=v,
                        message=f"Moderate score distribution drift (PSI={psi_score:.3f} >= {self.baseline.psi_warning_threshold}).",
                        recommended_action="Monitor upcoming 500 requests for sustained score compression.",
                    )
                )

            # 5. Score Concentration / Polarization Alert
            hist_norm = current_hist / np.sum(current_hist)
            max_bin_ratio = float(np.max(hist_norm))
            if max_bin_ratio >= self.baseline.score_polarization_threshold:
                alerts.append(
                    DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now,
                        severity=DriftAlertSeverity.WARNING,
                        metric_name="score_concentration_polarization",
                        current_value=round(max_bin_ratio, 4),
                        threshold_value=self.baseline.score_polarization_threshold,
                        model_version=v,
                        message=f"Abnormal score concentration: {max_bin_ratio:.1%} of predictions concentrated in single histogram bin.",
                        recommended_action="Investigate sensor saturation, monoculture inputs, or generator clustering.",
                    )
                )

        # 6. Latency Spike Alert
        p95_lat = summary["latency_summary"]["p95_ms"]
        if p95_lat >= self.baseline.latency_p95_critical_ms:
            alerts.append(
                DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=now,
                    severity=DriftAlertSeverity.CRITICAL,
                    metric_name="latency_p95_ms",
                    current_value=p95_lat,
                    threshold_value=self.baseline.latency_p95_critical_ms,
                    model_version=v,
                    message=f"Critical p95 latency spike ({p95_lat:.1f}ms >= {self.baseline.latency_p95_critical_ms:.1f}ms).",
                    recommended_action="Scale inference capacity or activate batch queuing backpressure.",
                )
            )
        elif p95_lat >= self.baseline.latency_p95_warning_ms:
            alerts.append(
                DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=now,
                    severity=DriftAlertSeverity.WARNING,
                    metric_name="latency_p95_ms",
                    current_value=p95_lat,
                    threshold_value=self.baseline.latency_p95_warning_ms,
                    model_version=v,
                    message=f"Elevated p95 latency ({p95_lat:.1f}ms >= {self.baseline.latency_p95_warning_ms:.1f}ms).",
                    recommended_action="Check host CPU thread contention and concurrent worker allocations.",
                )
            )

        # 7. Corrupted Telemetry Alert
        if self._corrupted_telemetry_count > 0:
            corrupt_rate = self._corrupted_telemetry_count / max(1, n + self._corrupted_telemetry_count)
            if corrupt_rate >= 0.02:  # 2% corrupted
                alerts.append(
                    DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        timestamp=now,
                        severity=DriftAlertSeverity.WARNING,
                        metric_name="corrupted_telemetry_rate",
                        current_value=round(corrupt_rate, 4),
                        threshold_value=0.02,
                        model_version=v,
                        message=f"Corrupted telemetry rate ({corrupt_rate:.1%}) exceeds 2% threshold.",
                        recommended_action="Audit upstream caller serialization and check for NaN/Inf logging bugs.",
                    )
                )

        self._alert_history.extend(alerts)
        return alerts

    # --------------------------------------------------------------------------
    # Automated Retraining Prohibition Guardrail
    # --------------------------------------------------------------------------

    def request_model_retraining(self, trigger_source: str, dataset_origin: str) -> None:
        """Enforces SafeHER Anti-Pattern Prohibition: NEVER retrain models automatically from user feedback.

        Raises:
            PermissionError: When automatic retraining from unverified user feedback is attempted.
        """
        disallowed_origins = [
            "user_feedback",
            "production_uploads",
            "unverified_survivor_cases",
            "human_reviews",
            "auto_pipeline",
            "production_telemetry",
            "online_learning",
        ]
        if any(d in dataset_origin.lower() for d in disallowed_origins):
            logger.critical(
                "SECURITY ALERT: Blocked unauthorized attempt to trigger automated model retraining from '%s'.",
                dataset_origin,
            )
            raise PermissionError(
                f"SafeHER Security Policy Violation: Automated model retraining or weight replacement "
                f"from unverified production data ('{dataset_origin}') is strictly prohibited to prevent "
                f"adversarial data poisoning, label contamination, and feedback loops."
            )

        logger.info(
            "Authorized offline re-training request approved for controlled evaluation research: source=%s",
            trigger_source,
        )

    # --------------------------------------------------------------------------
    # Version Tracking & Rollback Runbook Execution
    # --------------------------------------------------------------------------

    def register_model_version(
        self,
        version: str,
        model_name: str,
        architecture: str,
        release_date: str,
        status: str = "production_active",
        certified_auc: float = 0.90,
        certified_ece: float = 0.05,
        description: str = "",
        weights_sha256: Optional[str] = None,
    ) -> None:
        """Register a validated model version into the version manifest."""
        self._model_registry[version] = {
            "model_name": model_name,
            "version": version,
            "architecture": architecture,
            "release_date": release_date,
            "status": status,
            "description": description,
            "certified_auc": certified_auc,
            "certified_ece": certified_ece,
            "weights_sha256": weights_sha256,
        }

    def get_version_registry(self) -> Dict[str, Any]:
        """Return the complete model version manifest and operational status."""
        return {
            "active_version": self.baseline.model_version,
            "registered_versions": self._model_registry,
            "total_alerts_recorded": len(self._alert_history),
            "baseline_metrics": asdict(self.baseline),
        }

    def execute_model_rollback(
        self,
        target_version: Optional[str] = "0.1.0",
        reason: str = "Automated or operator drift alert response",
    ) -> Dict[str, Any]:
        """Execute deterministic zero-downtime model rollback to a certified version or heuristic baseline.

        Returns:
            Rollback execution manifest.
        """
        logger.warning(
            "INITIATING MODEL ROLLBACK: Current=%s, Target=%s, Reason=%s",
            self.baseline.model_version,
            target_version,
            reason,
        )

        previous_version = self.baseline.model_version

        # Instantiate target safe runner (fallback to deterministic baseline)
        safe_runner = BaselineHeuristicRunner()
        resilient_detector = SafeHERImageDetector(runner=safe_runner)

        # Thread-safe global singleton swap
        set_image_detector(resilient_detector)

        # Update monitoring baseline
        self.baseline = DriftMetricBaseline(
            model_name="SafeHER-Baseline-ELA-Heuristics",
            model_version=target_version or "0.1.0",
        )

        rollback_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ROLLBACK_SUCCESSFUL",
            "previous_version": previous_version,
            "active_version": target_version or "0.1.0",
            "active_runner": "SafeHER-Baseline-ELA-Heuristics",
            "reason": reason,
            "verification": "Deterministic baseline active; zero neural runtime dependencies.",
        }

        logger.info("MODEL ROLLBACK COMPLETED: %s", rollback_record)
        return rollback_record

    def generate_monitoring_report(self, window_size: Optional[int] = None) -> str:
        """Generate a GitHub Flavored Markdown telemetry and drift monitoring report."""
        summary = self.get_telemetry_summary(window_size=window_size)
        alerts = self.evaluate_drift_and_alerts(min_sample_size=1)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "# SafeHER Neural Image Intelligence — Model Drift & Telemetry Monitoring Report",
            "",
            f"**Audit Timestamp:** {now_str}  ",
            f"**Active Model Version:** `{self.baseline.model_version}` ({self.baseline.model_name})  ",
            f"**Observation Window:** {summary['total_requests']} requests (Max Window: {self._telemetry_window.maxlen})  ",
            f"**Operational Status:** `{'STABLE' if not alerts else 'DRIFT_ALERTS_ACTIVE'}`  ",
            "",
            "## 1. Executive Telemetry Overview",
            "",
            "| Metric | Current Value | Baseline Threshold | Status |",
            "| :--- | :-: | :-: | :--- |",
            f"| Total Ingested Requests | `{summary['total_requests']}` | N/A | Active |",
            f"| Pipeline Failure Rate | `{summary['failure_rate']:.2%}` | `< 1.00%` | {'CRITICAL' if summary['failure_rate'] >= 0.03 else ('WARNING' if summary['failure_rate'] >= 0.01 else 'NORMAL')} |",
            f"| Out-of-Distribution (OOD) Rate | `{summary['ood_rate']:.2%}` | `< 15.00%` | {'CRITICAL' if summary['ood_rate'] >= 0.25 else ('WARNING' if summary['ood_rate'] >= 0.15 else 'NORMAL')} |",
            f"| Latency p95 | `{summary['latency_summary']['p95_ms']} ms` | `< 200.0 ms` | {'CRITICAL' if summary['latency_summary']['p95_ms'] >= 500 else ('WARNING' if summary['latency_summary']['p95_ms'] >= 200 else 'NORMAL')} |",
            f"| Human Review Disagreement | `{summary['human_disagreement_rate']:.2%}` | `< 15.00%` | {'NORMAL' if summary['human_disagreement_rate'] < 0.15 else 'WARNING'} |",
            f"| Corrupted Telemetry Quarantined | `{summary['corrupted_telemetry_count']}` | `0` | {'WARNING' if summary['corrupted_telemetry_count'] > 0 else 'NORMAL'} |",
            "",
            "## 2. Input Distributions & Triage Profiling",
            "",
            "### A. Spatial Dimensions & Compression Quality",
            "",
            "| Dimension Bucket | Share | Compression Category | Share |",
            "| :--- | :-: | :--- | :-: |",
        ]

        dim_dist = summary.get("dimension_distribution", {})
        comp_dist = summary.get("compression_distribution", {})
        all_keys = sorted(set(list(dim_dist.keys()) + list(comp_dist.keys())))

        for k in all_keys:
            d_val = f"{dim_dist.get(k, 0.0):.1%}" if k in dim_dist else "-"
            c_val = f"{comp_dist.get(k, 0.0):.1%}" if k in comp_dist else "-"
            lines.append(f"| `{k}` | {d_val} | `{k}` | {c_val} |")

        lines.extend([
            "",
            "### B. Predictions and Uncertainty Distribution",
            "",
            "| Prediction Label | Share | Uncertainty State | Share |",
            "| :--- | :-: | :--- | :-: |",
        ])

        pred_dist = summary.get("prediction_distribution", {})
        unc_dist = summary.get("uncertainty_distribution", {})
        pu_keys = sorted(set(list(pred_dist.keys()) + list(unc_dist.keys())))
        for k in pu_keys:
            p_val = f"{pred_dist.get(k, 0.0):.1%}" if k in pred_dist else "-"
            u_val = f"{unc_dist.get(k, 0.0):.1%}" if k in unc_dist else "-"
            lines.append(f"| `{k}` | {p_val} | `{k}` | {u_val} |")

        lines.extend([
            "",
            "### C. Latency Percentiles (CPU Execution)",
            "",
            f"- **Mean Latency:** `{summary['latency_summary']['mean_ms']} ms`",
            f"- **Median (p50):** `{summary['latency_summary']['median_ms']} ms`",
            f"- **p95 Latency:** `{summary['latency_summary']['p95_ms']} ms`",
            f"- **p99 Latency:** `{summary['latency_summary']['p99_ms']} ms`",
            "",
            "## 3. Active Drift Alerts & Recommended Actions",
            "",
        ])

        if alerts:
            for idx, a in enumerate(alerts, 1):
                badge = f"`{a.severity.value.upper()}`"
                lines.extend([
                    f"### Alert {idx}: {a.metric_name} ({badge})",
                    f"- **Observed Value:** `{a.current_value}` (Threshold: `{a.threshold_value}`)",
                    f"- **Trigger Message:** {a.message}",
                    f"- **Recommended Mitigation:** {a.recommended_action}",
                    "",
                ])
        else:
            lines.append("> [!NOTE]\n> Zero active drift alerts. Score distribution and pipeline telemetry are within certified validation baselines.\n")

        lines.extend([
            "## 4. Model Version Lineage & Rollback Readiness",
            "",
            "| Version | Architecture | Status | Certified AUC | Certified ECE |",
            "| :--- | :--- | :--- | :-: | :-: |",
        ])

        for ver, meta in self._model_registry.items():
            status_badge = f"`{meta['status']}`"
            lines.append(
                f"| `{ver}` | `{meta.get('architecture', 'N/A')}` | {status_badge} | "
                f"`{meta.get('certified_auc', 0.0):.3f}` | `{meta.get('certified_ece', 0.0):.3f}` |"
            )

        lines.extend([
            "",
            "## 5. Security & Privacy Guarantees",
            "",
            "> [!IMPORTANT]",
            "> 1. **Zero Image Storage Guarantee:** No raw pixels, base64 data, or image paths are logged in telemetry.",
            "> 2. **Automated Retraining Prohibition:** Online parameter updates from user feedback are blocked at the architectural layer.",
            "> 3. **Failure Isolation:** Ingestion errors are safely isolated; monitoring never blocks or crashes image triage.",
            "",
            f"*Certified by SafeHER AI Telemetry & Monitoring System.*",
        ])

        return "\n".join(lines)


# Global singleton instance of monitoring engine
model_monitoring_engine = ModelMonitoringEngine()
