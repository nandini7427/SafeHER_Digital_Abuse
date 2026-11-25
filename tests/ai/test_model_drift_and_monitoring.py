"""Automated Verification Suite for Privacy-Preserving Model Drift & Telemetry Monitoring.

Verifies SafeHER Milestone 3 monitoring requirements:
1. Privacy-Safe Telemetry Aggregation:
   - Ingests inference metrics without retaining raw image pixels, base64 strings, or file paths.
2. Population Stability Index (PSI) Covariate Shift Detection:
   - Computes statistical drift across forensic score distributions (stable < 0.1, moderate < 0.25, critical >= 0.25).
3. Drift Indicators & Multi-Tier Alert Conditions:
   - Inference failure rate alerts (Warning @ 1%, Critical @ 3%).
   - Out-of-distribution (OOD) rate alerts (Warning @ 15%, Critical @ 25%).
   - Human review disagreement rate alerts (Warning @ 15%, Critical @ 25%).
   - Score distribution PSI drift alerts.
4. Anti-Pattern Prohibition Guardrail:
   - Programmatic block preventing automated retraining from unverified user feedback or human reviews.
5. Pinned Model Version Registry & Deterministic Rollback:
   - Immutable version tracking and instant zero-downtime rollback runbook execution.
"""

from datetime import datetime, timezone
import io
from typing import List
import uuid

import numpy as np
from PIL import Image, ImageDraw
import pytest

from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.monitoring import (
    DriftAlert,
    DriftAlertSeverity,
    DriftMetricBaseline,
    ModelMonitoringEngine,
    TelemetryRecord,
    compute_population_stability_index,
)
from ai_core.image_intelligence.schemas import (
    AssessmentLabel,
    DetectionTask,
    ImageAssessmentResponse,
    ImageMetadataSummary,
    ImageQualityIndicators,
    UncertaintyState,
)
from ai_core.image_intelligence.service import get_image_detector, set_image_detector


def _create_mock_response(
    score: float = 0.20,
    uncertainty: UncertaintyState = UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
    label: AssessmentLabel = AssessmentLabel.NO_OBVIOUS_ARTIFACTS,
    is_ood: bool = False,
) -> ImageAssessmentResponse:
    """Helper to generate mock ImageAssessmentResponse for telemetry testing."""
    return ImageAssessmentResponse(
        assessment_id=uuid.uuid4(),
        detector_name="SafeHER-Baseline-ELA-Heuristics",
        detector_version="0.2.0",
        detection_task=DetectionTask.HYBRID_MANIPULATION,
        raw_model_score=score,
        is_calibrated=False,
        assessment_label=label,
        uncertainty_state=UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN if is_ood else uncertainty,
        explanation="Test assessment output",
        quality_indicators=ImageQualityIndicators(
            width=512,
            height=512,
            megapixels=0.26,
            aspect_ratio=1.0,
            resolution_ok=True,
            blur_score=150.0,
            is_blurry=False,
            compression_artifact_score=0.15,
            has_heavy_compression=False,
            mean_luminance=120.0,
            overexposed_ratio=0.01,
            underexposed_ratio=0.01,
            exposure_level="normal",
            is_monochromatic_or_flat=False,
            is_screen_capture=False,
            overall_quality_score=0.92,
        ),
        metadata_summary=ImageMetadataSummary(
            format="JPEG",
            dimensions=(512, 512),
            size_bytes=45000,
            sha256="a" * 64,
            has_exif=True,
            sanitized_exif={"Make": "TestCam", "Model": "TestModel"},
            color_mode="RGB",
        ),
    )


def test_telemetry_ingestion_privacy_guarantee():
    """Verify that telemetry records capture categorical metrics without storing image pixels or base64."""
    engine = ModelMonitoringEngine()
    resp = _create_mock_response(score=0.25)

    engine.record_assessment_telemetry(
        response=resp,
        is_failure=False,
        elapsed_ms=120.5,
        image_format="JPEG",
        dimensions=(512, 512),
    )

    summary = engine.get_telemetry_summary()
    assert summary["total_requests"] == 1
    assert summary["failure_rate"] == 0.0
    assert summary["ood_rate"] == 0.0
    assert summary["mean_score"] == 0.25
    assert summary["format_distribution"]["JPEG"] == 1.0
    assert summary["dimension_distribution"]["standard"] == 1.0

    # Inspect internal records for zero raw image data
    record = engine._telemetry_window[0]
    record_dict = record.__dict__
    for key, val in record_dict.items():
        assert "base64" not in str(val).lower()
        assert "data:image" not in str(val).lower()
        assert not isinstance(val, (bytes, bytearray))


def test_population_stability_index_computation():
    """Verify PSI calculation for identical, slightly shifted, and radically drifted distributions."""
    baseline = np.array([0.35, 0.25, 0.12, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02, 0.03])

    # 1. Identical distribution -> PSI should be very close to 0.0
    identical_psi = compute_population_stability_index(baseline, baseline)
    assert identical_psi < 0.001, f"Identical distributions must have near-zero PSI; got {identical_psi}"

    # 2. Minor perturbation -> PSI should remain under 0.10
    minor_shift = np.array([0.33, 0.26, 0.13, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03])
    minor_psi = compute_population_stability_index(baseline, minor_shift)
    assert minor_psi < 0.10, f"Minor shift should have PSI < 0.10; got {minor_psi}"

    # 3. Radically inverted distribution (high scores dominate) -> PSI >= 0.25
    drastic_shift = np.array([0.03, 0.02, 0.03, 0.03, 0.04, 0.05, 0.08, 0.12, 0.25, 0.35])
    drastic_psi = compute_population_stability_index(baseline, drastic_shift)
    assert drastic_psi >= 0.25, f"Drastic shift must trigger PSI >= 0.25; got {drastic_psi}"


def test_alert_conditions_failure_rate_thresholds():
    """Verify that rising inference failure rates trigger Warning and Critical alerts."""
    engine = ModelMonitoringEngine()

    # 50 successful requests
    for _ in range(50):
        engine.record_assessment_telemetry(response=_create_mock_response(), is_failure=False)

    # Add 1 failure -> 1/51 = ~1.96% failure rate (exceeds 1.0% Warning threshold)
    engine.record_assessment_telemetry(response=None, is_failure=True, failure_category="Timeout")
    alerts_warning = engine.evaluate_drift_and_alerts(min_sample_size=20)
    warning_alerts = [a for a in alerts_warning if a.metric_name == "inference_failure_rate"]
    assert len(warning_alerts) == 1
    assert warning_alerts[0].severity == DriftAlertSeverity.WARNING

    # Add 2 more failures -> 3/53 = ~5.66% failure rate (exceeds 3.0% Critical threshold)
    engine.record_assessment_telemetry(response=None, is_failure=True, failure_category="Timeout")
    engine.record_assessment_telemetry(response=None, is_failure=True, failure_category="Crash")
    alerts_critical = engine.evaluate_drift_and_alerts(min_sample_size=20)
    critical_alerts = [
        a for a in alerts_critical
        if a.metric_name == "inference_failure_rate" and a.severity == DriftAlertSeverity.CRITICAL
    ]
    assert len(critical_alerts) >= 1
    assert "rollback" in critical_alerts[0].recommended_action.lower()


def test_alert_conditions_ood_spike():
    """Verify that a surge in out-of-distribution traffic triggers OOD drift alerts."""
    engine = ModelMonitoringEngine()

    # 30 standard in-distribution images
    for _ in range(30):
        engine.record_assessment_telemetry(response=_create_mock_response(is_ood=False))

    # Add 10 OOD images -> 10/40 = 25% OOD rate (exceeds 15% Warning and 25% Critical threshold)
    for _ in range(10):
        engine.record_assessment_telemetry(response=_create_mock_response(is_ood=True))

    alerts = engine.evaluate_drift_and_alerts(min_sample_size=20)
    ood_alerts = [a for a in alerts if a.metric_name == "out_of_distribution_rate"]
    assert len(ood_alerts) >= 1
    assert any(a.severity == DriftAlertSeverity.CRITICAL for a in ood_alerts)


def test_alert_conditions_human_review_disagreement_spike():
    """Verify that human-AI review disagreement spikes trigger concept drift alerts."""
    engine = ModelMonitoringEngine()

    # Generate baseline traffic
    for _ in range(30):
        engine.record_assessment_telemetry(response=_create_mock_response())

    # Record 20 human reviews: 6 disagreements (30% disagreement rate >= 25% Critical threshold)
    for _ in range(14):
        engine.record_human_review_metric(is_disagreement=False, human_verdict="concurs_with_ai")
    for _ in range(6):
        engine.record_human_review_metric(is_disagreement=True, human_verdict="disagrees_suspicious_manipulation")

    summary = engine.get_telemetry_summary()
    assert summary["human_disagreement_rate"] == 0.30

    alerts = engine.evaluate_drift_and_alerts(min_sample_size=20)
    disagreement_alerts = [a for a in alerts if a.metric_name == "human_review_disagreement_rate"]
    assert len(disagreement_alerts) >= 1
    assert any(a.severity == DriftAlertSeverity.CRITICAL for a in disagreement_alerts)


def test_alert_conditions_psi_drift_detection():
    """Verify that score distribution skewing triggers Population Stability Index alerts."""
    engine = ModelMonitoringEngine()

    # Populate window with scores heavily concentrated at the high end (0.8 - 1.0)
    # whereas baseline reference expects low scores (0.0 - 0.2)
    for _ in range(40):
        engine.record_assessment_telemetry(response=_create_mock_response(score=0.92))

    alerts = engine.evaluate_drift_and_alerts(min_sample_size=20)
    psi_alerts = [a for a in alerts if a.metric_name == "population_stability_index"]
    assert len(psi_alerts) >= 1
    assert psi_alerts[0].current_value >= 0.25
    assert psi_alerts[0].severity == DriftAlertSeverity.CRITICAL


def test_automated_retraining_prohibition_guardrail():
    """Verify that attempting to trigger automatic online model retraining is strictly prohibited."""
    engine = ModelMonitoringEngine()

    # Attempting to trigger automated model retraining from user feedback MUST fail
    with pytest.raises(PermissionError) as exc_user:
        engine.request_model_retraining(
            trigger_source="auto_pipeline",
            dataset_origin="user_feedback",
        )
    assert "automated model retraining" in str(exc_user.value).lower()
    assert "unverified production data" in str(exc_user.value).lower()

    # Attempting from human reviews MUST also fail
    with pytest.raises(PermissionError) as exc_rev:
        engine.request_model_retraining(
            trigger_source="reviewer_hub",
            dataset_origin="human_reviews",
        )
    assert "strictly prohibited" in str(exc_rev.value).lower()

    # Controlled offline research dataset should pass
    engine.request_model_retraining(
        trigger_source="offline_research_benchmark",
        dataset_origin="controlled_certified_benchmark_v3",
    )


def test_model_version_registry_and_tracking():
    """Verify version manifest tracks semantic versioning, release history, and baseline metrics."""
    engine = ModelMonitoringEngine()
    registry = engine.get_version_registry()

    assert registry["active_version"] == "0.2.0"
    assert "0.2.0" in registry["registered_versions"]
    assert "0.1.0" in registry["registered_versions"]

    active_meta = registry["registered_versions"]["0.2.0"]
    assert active_meta["status"] == "production_active"
    assert active_meta["certified_auc"] > 0.85
    assert registry["baseline_metrics"]["baseline_auc"] == 0.90


def test_deterministic_model_rollback_execution():
    """Verify instant, zero-downtime rollback to a certified version or deterministic baseline."""
    engine = ModelMonitoringEngine()

    # Execute rollback
    result = engine.execute_model_rollback(
        target_version="0.1.0",
        reason="Drift test trigger: critical PSI shift",
    )

    assert result["status"] == "ROLLBACK_SUCCESSFUL"
    assert result["previous_version"] == "0.2.0"
    assert result["active_version"] == "0.1.0"
    assert result["active_runner"] == "SafeHER-Baseline-ELA-Heuristics"

    # Verify detector singleton was safely swapped to the baseline
    detector = get_image_detector()
    assert isinstance(detector, SafeHERImageDetector)

    # Perform inference to confirm active detector works smoothly post-rollback
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    resp = detector.detect(buf.getvalue())
    assert isinstance(resp, ImageAssessmentResponse)
    assert "Baseline" in resp.detector_name


# ============================================================================
# Milestone 4 Neural Detector Drift & Telemetry Tests
# ============================================================================

def test_neural_telemetry_ingestion_and_privacy():
    """Verify that neural telemetry ingests model metrics with zero image or PII retention."""
    engine = ModelMonitoringEngine()

    engine.record_neural_telemetry(
        model_name="SafeHER-Model-A-Neural-Detector",
        model_version="1.0.0-m4.3",
        architecture="mobilenet_v3_large",
        raw_score=0.824,
        calibrated_score=0.791,
        predicted_class="AI_GENERATED",
        uncertainty_state="USABLE_EVIDENCE",
        is_ood=False,
        dimensions=(1024, 1024),
        image_format="PNG",
        compression_category="pristine_uncompressed",
        quality_blur=False,
        quality_exposure=False,
        quality_resolution_ok=True,
        is_failure=False,
        elapsed_ms=45.2,
        neural_status="active",
    )

    summary = engine.get_telemetry_summary()
    assert summary["total_requests"] == 1
    assert summary["prediction_distribution"]["AI_GENERATED"] == 1.0
    assert summary["uncertainty_distribution"]["USABLE_EVIDENCE"] == 1.0
    assert summary["dimension_distribution"]["standard"] == 1.0
    assert summary["compression_distribution"]["pristine_uncompressed"] == 1.0
    assert summary["model_version_usage"]["1.0.0-m4.3"] == 1
    assert summary["calibration_summary"]["calibrated_predictions_count"] == 1
    assert summary["calibration_summary"]["mean_calibrated_confidence"] == 0.791

    # Verify zero raw pixels or base64 data
    rec = engine._telemetry_window[0]
    for k, v in rec.__dict__.items():
        assert "base64" not in str(v).lower()
        assert "data:image" not in str(v).lower()
        assert not isinstance(v, (bytes, bytearray))


def test_dimension_and_compression_categorization():
    """Verify dimension bucketing (thumbnail, standard, hd, large) and compression categorization."""
    engine = ModelMonitoringEngine()

    # 1. Thumbnail (<= 256)
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", dimensions=(128, 128),
        image_format="PNG", elapsed_ms=10.0,
    )
    # 2. Standard (257 - 1080)
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", dimensions=(800, 600),
        image_format="JPEG", elapsed_ms=15.0,
    )
    # 3. HD (1081 - 1920)
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", dimensions=(1920, 1080),
        image_format="WEBP", elapsed_ms=25.0,
    )
    # 4. Large (> 1920)
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", dimensions=(3840, 2160),
        image_format="JPEG", elapsed_ms=50.0,
    )

    summary = engine.get_telemetry_summary()
    dims = summary["dimension_distribution"]
    assert dims["thumbnail"] == 0.25
    assert dims["standard"] == 0.25
    assert dims["hd"] == 0.25
    assert dims["large"] == 0.25


def test_latency_distribution_percentiles():
    """Verify mean, median (p50), p95, and p99 latency calculation."""
    engine = ModelMonitoringEngine()

    # Ingest 100 requests with known latencies: 1ms through 100ms
    for ms in range(1, 101):
        engine.record_neural_telemetry(
            model_name="ModelA", model_version="1.0.0", elapsed_ms=float(ms),
        )

    summary = engine.get_telemetry_summary()
    lat = summary["latency_summary"]
    assert lat["mean_ms"] == 50.5
    assert lat["median_ms"] == 50.5
    assert lat["p95_ms"] == 95.05
    assert lat["p99_ms"] == 99.01


def test_latency_spike_alerts():
    """Verify that p95 latency degradation triggers warning and critical alerts."""
    engine = ModelMonitoringEngine()

    # 30 fast requests (50ms)
    for _ in range(30):
        engine.record_neural_telemetry(model_name="ModelA", model_version="1.0.0", elapsed_ms=50.0)

    # Add high-latency requests pushing p95 above 200ms Warning threshold
    for _ in range(10):
        engine.record_neural_telemetry(model_name="ModelA", model_version="1.0.0", elapsed_ms=250.0)

    alerts_warn = engine.evaluate_drift_and_alerts(min_sample_size=20)
    lat_warn = [a for a in alerts_warn if a.metric_name == "latency_p95_ms"]
    assert len(lat_warn) >= 1
    assert any(a.severity == DriftAlertSeverity.WARNING for a in lat_warn)

    # Push p95 above 500ms Critical threshold
    for _ in range(20):
        engine.record_neural_telemetry(model_name="ModelA", model_version="1.0.0", elapsed_ms=600.0)

    alerts_crit = engine.evaluate_drift_and_alerts(min_sample_size=20)
    lat_crit = [a for a in alerts_crit if a.metric_name == "latency_p95_ms" and a.severity == DriftAlertSeverity.CRITICAL]
    assert len(lat_crit) >= 1
    assert "capacity" in lat_crit[0].recommended_action.lower()


def test_score_concentration_polarization_alert():
    """Verify that abnormal concentration in a single score bin triggers a polarization alert."""
    engine = ModelMonitoringEngine()

    # Ingest 30 requests where 25 (83.3% > 70% threshold) fall in the top bin (0.9 - 1.0)
    for _ in range(25):
        engine.record_neural_telemetry(model_name="ModelA", model_version="1.0.0", raw_score=0.95)
    for _ in range(5):
        engine.record_neural_telemetry(model_name="ModelA", model_version="1.0.0", raw_score=0.20)

    alerts = engine.evaluate_drift_and_alerts(min_sample_size=20)
    polar_alerts = [a for a in alerts if a.metric_name == "score_concentration_polarization"]
    assert len(polar_alerts) >= 1
    assert polar_alerts[0].current_value >= 0.70
    assert "monoculture" in polar_alerts[0].recommended_action.lower()


def test_corrupted_telemetry_isolation_and_alerts():
    """Verify that corrupt telemetry (NaN, Inf, invalid types) is quarantined without raising exceptions."""
    engine = ModelMonitoringEngine()

    # 1. Non-finite raw_score (NaN)
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", raw_score=float("nan"),
    )
    # 2. Non-finite raw_score (Inf)
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", raw_score=float("inf"),
    )
    # 3. Invalid string as score
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", raw_score="corrupted_not_a_number",  # type: ignore
    )
    # 4. Valid record
    engine.record_neural_telemetry(
        model_name="ModelA", model_version="1.0.0", raw_score=0.55,
    )

    assert engine._corrupted_telemetry_count >= 3
    summary = engine.get_telemetry_summary()
    assert summary["corrupted_telemetry_count"] >= 3
    # The valid record must be recorded properly
    assert summary["total_requests"] == 4

    # Test alert when corrupted rate exceeds 2%
    alerts = engine.evaluate_drift_and_alerts(min_sample_size=1)
    corrupt_alerts = [a for a in alerts if a.metric_name == "corrupted_telemetry_rate"]
    assert len(corrupt_alerts) >= 1
    assert corrupt_alerts[0].severity == DriftAlertSeverity.WARNING


def test_missing_metrics_graceful_handling():
    """Verify that passing None or empty fields does not cause runtime crashes."""
    engine = ModelMonitoringEngine()

    engine.record_neural_telemetry(
        model_name="ModelA",
        model_version="1.0.0",
        raw_score=None,
        calibrated_score=None,
        predicted_class=None,
        uncertainty_state=None,
        dimensions=(0, 0),
        image_format="UNKNOWN",
        elapsed_ms=0.0,
    )

    summary = engine.get_telemetry_summary()
    assert summary["total_requests"] == 1
    assert summary["failure_rate"] == 0.0
    assert summary["mean_score"] == 0.0


def test_model_version_registration_and_usage_tracking():
    """Verify registering neural model versions and tracking invocation counts across versions."""
    engine = ModelMonitoringEngine()

    engine.register_model_version(
        version="1.1.0-canary",
        model_name="SafeHER-Model-A-Experimental",
        architecture="efficientnet_b0",
        release_date="2026-11-25",
        status="canary",
        certified_auc=0.948,
        certified_ece=0.038,
        weights_sha256="c" * 64,
    )

    # Ingest 3 requests with production version, 1 request with canary version
    for _ in range(3):
        engine.record_neural_telemetry(model_name="ModelA", model_version="1.0.0-m4.3")
    engine.record_neural_telemetry(model_name="ModelA", model_version="1.1.0-canary")

    summary = engine.get_telemetry_summary()
    usage = summary["model_version_usage"]
    assert usage["1.0.0-m4.3"] == 3
    assert usage["1.1.0-canary"] == 1

    registry = engine.get_version_registry()
    assert "1.1.0-canary" in registry["registered_versions"]
    assert registry["registered_versions"]["1.1.0-canary"]["status"] == "canary"


def test_monitoring_report_generation():
    """Verify that generate_monitoring_report renders publication-quality markdown."""
    engine = ModelMonitoringEngine()

    for i in range(25):
        engine.record_neural_telemetry(
            model_name="SafeHER-Model-A-Neural-Detector",
            model_version="1.0.0-m4.3",
            raw_score=0.10 + (i % 5) * 0.15,
            calibrated_score=0.12 + (i % 5) * 0.14,
            predicted_class="CAMERA_PHOTOGRAPH" if i % 2 == 0 else "AI_GENERATED",
            uncertainty_state="USABLE_EVIDENCE",
            dimensions=(1024, 768),
            image_format="JPEG",
            elapsed_ms=35.0 + i,
        )

    report_md = engine.generate_monitoring_report()
    assert "# SafeHER Neural Image Intelligence — Model Drift & Telemetry Monitoring Report" in report_md
    assert "Executive Telemetry Overview" in report_md
    assert "Input Distributions & Triage Profiling" in report_md
    assert "Spatial Dimensions & Compression Quality" in report_md
    assert "Predictions and Uncertainty Distribution" in report_md
    assert "Latency Percentiles" in report_md
    assert "Model Version Lineage & Rollback Readiness" in report_md
    assert "Zero Image Storage Guarantee" in report_md
