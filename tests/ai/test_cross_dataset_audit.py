"""Unit tests for SafeHER Cross-Dataset Generalization Audit modules."""

import pytest

from ai_core.evaluation.cross_dataset_benchmark import (
    build_authentic_stress_dataset,
    build_in_distribution_dataset,
    build_pure_ai_generated_dataset,
    build_unseen_manipulation_dataset,
    build_wild_social_media_dataset,
)
from ai_core.evaluation.run_cross_dataset_audit import (
    CrossDatasetMetrics,
    evaluate_dataset,
    format_cross_dataset_report,
)
from ai_core.image_intelligence.detector import SafeHERImageDetector


def test_cross_dataset_sample_generation():
    """Verify all 5 dataset builders return valid non-empty samples."""
    id_ds = build_in_distribution_dataset()
    assert len(id_ds) == 50
    assert any(s.is_manipulated for s in id_ds)
    assert any(not s.is_manipulated for s in id_ds)

    wild_ds = build_wild_social_media_dataset(count=10)
    assert len(wild_ds) == 10
    assert any(s.is_manipulated for s in wild_ds)
    assert any(not s.is_manipulated for s in wild_ds)

    unseen_ds = build_unseen_manipulation_dataset(count=10)
    assert len(unseen_ds) == 10

    aigen_ds = build_pure_ai_generated_dataset(count=10)
    assert len(aigen_ds) == 10

    stress_ds = build_authentic_stress_dataset()
    assert len(stress_ds) == 40
    # All stress samples must be authentic
    assert all(not s.is_manipulated for s in stress_ds)


def test_unseen_manipulation_techniques():
    """Verify unseen manipulation dataset includes Poisson, diffusion, anti-forensics, and face swap."""
    unseen_ds = build_unseen_manipulation_dataset(count=40)
    manip_types = {s.technique for s in unseen_ds if s.is_manipulated}

    assert "poisson_seamless_clone" in manip_types
    assert "diffusion_inpaint" in manip_types
    assert "adversarial_noise_dither" in manip_types
    assert "deepfake_face_swap" in manip_types


def test_evaluate_dataset_metrics():
    """Verify evaluate_dataset computes valid metrics and handles zero division."""
    detector = SafeHERImageDetector()
    mini_ds = build_unseen_manipulation_dataset(count=4)

    metrics = evaluate_dataset("test_mini_unseen", mini_ds, detector)

    assert metrics.sample_count == 4
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.balanced_accuracy <= 1.0
    assert 0.0 <= metrics.false_positive_rate <= 1.0
    assert 0.0 <= metrics.false_negative_rate <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_format_cross_dataset_report_contains_verdict():
    """Verify markdown report formatter outputs clear sections and research verdict."""
    dummy_metric = CrossDatasetMetrics(
        dataset_name="in_distribution_test",
        sample_count=50,
        positives_count=25,
        negatives_count=25,
        true_positives=0,
        false_positives=0,
        true_negatives=25,
        false_negatives=25,
        accuracy=0.5,
        balanced_accuracy=0.5,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        false_positive_rate=0.0,
        false_negative_rate=1.0,
        roc_auc=0.5,
        continuous_variance_auc=0.66,
        auth_mean_variance=0.37,
        manip_mean_variance=0.85,
    )

    report_md = format_cross_dataset_report([dummy_metric])
    assert "SafeHER Milestone 3 — Cross-Dataset Generalization and Domain Shift Audit" in report_md
    assert "RESEARCH VERDICT" in report_md
    assert "NOT SUFFICIENTLY RELIABLE" in report_md
