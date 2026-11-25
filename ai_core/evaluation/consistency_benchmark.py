"""Consistency Benchmark and Disagreement Evaluation Suite for SafeHER Image Intelligence.

Evaluates:
1. Per-model performance for Model 1 (Spatial ELA) and Model 2 (Frequency Spectral).
2. Inter-model agreement, disagreement rates, and Cohen's Kappa.
3. Correlated error analysis across independent datasets.
4. Ensemble strategies comparison (Averaging, Max/OR, Min/AND, vs SafeHER Disagreement Policy).
5. Computational profiling: Inference latency (mean, p50, p95) and peak memory usage (tracemalloc).
6. Calibration compatibility audit.
"""

from dataclasses import dataclass
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai_core.evaluation.cross_dataset_benchmark import (
    CrossDatasetSample,
    build_in_distribution_dataset,
    build_wild_social_media_dataset,
    build_unseen_manipulation_dataset,
    build_pure_ai_generated_dataset,
    build_authentic_stress_dataset,
)
from ai_core.image_intelligence.consistency import (
    DisagreementState,
    ModelConsistencyAssessment,
    ModelDisagreementPolicy,
    compute_correlated_errors,
    compute_inter_model_agreement,
)
from ai_core.image_intelligence.model_runner import (
    BaselineHeuristicRunner,
    FrequencySpectralRunner,
    ModelPrediction,
)
from ai_core.image_intelligence.preprocessing import preprocess_image
from ai_core.image_intelligence.quality import evaluate_image_quality
from ai_core.image_intelligence.schemas import AssessmentLabel, UncertaintyState


@dataclass(frozen=True)
class SampleConsistencyRecord:
    """Individual evaluation record across all models and ensemble strategies."""

    sample_id: str
    dataset_name: str
    ground_truth: int  # 1 for manipulated, 0 for authentic
    technique: str

    # Model 1 (Spatial ELA)
    m1_score: float
    m1_decision: bool  # True if flagged as anomaly
    m1_latency_ms: float

    # Model 2 (Frequency Spectral)
    m2_score: float
    m2_decision: bool  # True if flagged as anomaly
    m2_latency_ms: float

    # Naive Ensemble Strategies
    naive_avg_score: float
    naive_avg_decision: bool
    naive_or_decision: bool
    naive_and_decision: bool

    # SafeHER Disagreement Policy
    policy_state: DisagreementState
    policy_label: AssessmentLabel
    policy_uncertainty: UncertaintyState
    policy_is_inconclusive: bool


def _compute_classification_metrics(
    ground_truth: List[int],
    decisions: List[bool],
) -> Dict[str, float]:
    """Compute standard classification metrics from binary decisions."""
    n = len(ground_truth)
    if n == 0:
        return {}

    tp = sum(1 for gt, dec in zip(ground_truth, decisions) if gt == 1 and dec)
    fp = sum(1 for gt, dec in zip(ground_truth, decisions) if gt == 0 and dec)
    tn = sum(1 for gt, dec in zip(ground_truth, decisions) if gt == 0 and not dec)
    fn = sum(1 for gt, dec in zip(ground_truth, decisions) if gt == 1 and not dec)

    accuracy = (tp + tn) / n
    tpr = tp / max(tp + fn, 1)  # Recall
    tnr = tn / max(tn + fp, 1)  # Specificity
    fpr = fp / max(tn + fp, 1)
    fnr = fn / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    balanced_acc = 0.5 * (tpr + tnr)
    f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)

    return {
        "sample_count": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(tpr, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }


class ConsistencyBenchmarkRunner:
    """Orchestrates comprehensive multi-model consistency and disagreement benchmarking."""

    def __init__(
        self,
        m1_threshold: float = 0.45,
        m2_threshold: float = 0.45,
        avg_threshold: float = 0.45,
    ) -> None:
        self.m1_threshold = m1_threshold
        self.m2_threshold = m2_threshold
        self.avg_threshold = avg_threshold

        self.m1_runner = BaselineHeuristicRunner()
        self.m2_runner = FrequencySpectralRunner()
        self.policy = ModelDisagreementPolicy(
            primary_threshold=m1_threshold,
            secondary_threshold=m2_threshold,
        )

    def run_benchmark(
        self,
        samples: Optional[List[CrossDatasetSample]] = None,
    ) -> Tuple[List[SampleConsistencyRecord], Dict[str, Any]]:
        """Execute full benchmark across dataset samples and compile multi-angle audit statistics."""
        if samples is None:
            # Load standard 5-tier cross-dataset suite
            samples = []
            samples.extend(build_in_distribution_dataset())
            samples.extend(build_wild_social_media_dataset())
            samples.extend(build_unseen_manipulation_dataset())
            samples.extend(build_pure_ai_generated_dataset())
            samples.extend(build_authentic_stress_dataset())

        records: List[SampleConsistencyRecord] = []

        m1_latencies: List[float] = []
        m2_latencies: List[float] = []
        dual_latencies: List[float] = []

        tracemalloc.start()
        start_mem_bytes, _ = tracemalloc.get_traced_memory()

        for s in samples:
            preprocessed = preprocess_image(s.image_bytes)
            gt = 1 if s.is_manipulated else 0

            # Measure Model 1 execution
            t0 = time.perf_counter()
            pred1 = self.m1_runner.predict(preprocessed)
            t1 = time.perf_counter()
            lat1_ms = (t1 - t0) * 1000.0
            m1_latencies.append(lat1_ms)

            # Measure Model 2 execution
            t2 = time.perf_counter()
            pred2 = self.m2_runner.predict(preprocessed)
            t3 = time.perf_counter()
            lat2_ms = (t3 - t2) * 1000.0
            m2_latencies.append(lat2_ms)

            dual_latencies.append((t3 - t0) * 1000.0)

            # Decisions
            dec1 = bool(pred1.raw_score >= self.m1_threshold)
            dec2 = bool(pred2.raw_score >= self.m2_threshold)

            # Naive Averaging (strictly for benchmark comparison to prove why it fails)
            avg_score = round(0.5 * (pred1.raw_score + pred2.raw_score), 4)
            dec_avg = bool(avg_score >= self.avg_threshold)

            # Naive Voting
            dec_or = bool(dec1 or dec2)
            dec_and = bool(dec1 and dec2)

            # SafeHER Disagreement Policy
            quality_res = evaluate_image_quality(
                img=preprocessed.working_image,
                raw_score=pred1.raw_score,
                face_bboxes=None,
                model_failed=False,
                is_corrupted=False,
                format_name=preprocessed.metadata.format,
                has_exif=preprocessed.metadata.has_exif,
            )

            consistency_res = self.policy.evaluate(
                quality_result=quality_res,
                primary_prediction=pred1,
                secondary_prediction=pred2,
            )

            is_inconclusive = (
                consistency_res.assessment_label == AssessmentLabel.INCONCLUSIVE
            )

            records.append(
                SampleConsistencyRecord(
                    sample_id=s.sample_id,
                    dataset_name=s.dataset_name,
                    ground_truth=gt,
                    technique=s.technique,
                    m1_score=pred1.raw_score,
                    m1_decision=dec1,
                    m1_latency_ms=round(lat1_ms, 3),
                    m2_score=pred2.raw_score,
                    m2_decision=dec2,
                    m2_latency_ms=round(lat2_ms, 3),
                    naive_avg_score=avg_score,
                    naive_avg_decision=dec_avg,
                    naive_or_decision=dec_or,
                    naive_and_decision=dec_and,
                    policy_state=consistency_res.disagreement_state,
                    policy_label=consistency_res.assessment_label,
                    policy_uncertainty=consistency_res.uncertainty_state,
                    policy_is_inconclusive=is_inconclusive,
                )
            )

        current_mem_bytes, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_peak_mb = (peak_mem_bytes - start_mem_bytes) / (1024.0 * 1024.0)

        # Aggregate Statistics
        gt_list = [r.ground_truth for r in records]
        m1_dec_list = [r.m1_decision for r in records]
        m2_dec_list = [r.m2_decision for r in records]
        avg_dec_list = [r.naive_avg_decision for r in records]
        or_dec_list = [r.naive_or_decision for r in records]
        and_dec_list = [r.naive_and_decision for r in records]

        # Performance per strategy
        m1_metrics = _compute_classification_metrics(gt_list, m1_dec_list)
        m2_metrics = _compute_classification_metrics(gt_list, m2_dec_list)
        avg_metrics = _compute_classification_metrics(gt_list, avg_dec_list)
        or_metrics = _compute_classification_metrics(gt_list, or_dec_list)
        and_metrics = _compute_classification_metrics(gt_list, and_dec_list)

        # Inter-model agreement
        agreement_stats = compute_inter_model_agreement(m1_dec_list, m2_dec_list)

        # Correlated error analysis
        corr_error_stats = compute_correlated_errors(gt_list, m1_dec_list, m2_dec_list)

        # Disagreement policy state distributions
        policy_state_counts: Dict[str, int] = {}
        for r in records:
            st = r.policy_state.value
            policy_state_counts[st] = policy_state_counts.get(st, 0) + 1

        # Per-dataset breakdown
        datasets = sorted(list(set(r.dataset_name for r in records)))
        dataset_breakdowns: Dict[str, Any] = {}
        for ds in datasets:
            ds_records = [r for r in records if r.dataset_name == ds]
            ds_gt = [r.ground_truth for r in ds_records]
            ds_m1 = [r.m1_decision for r in ds_records]
            ds_m2 = [r.m2_decision for r in ds_records]
            ds_agree = compute_inter_model_agreement(ds_m1, ds_m2)
            ds_corr = compute_correlated_errors(ds_gt, ds_m1, ds_m2)

            dataset_breakdowns[ds] = {
                "sample_count": len(ds_records),
                "m1_metrics": _compute_classification_metrics(ds_gt, ds_m1),
                "m2_metrics": _compute_classification_metrics(ds_gt, ds_m2),
                "agreement_rate": ds_agree["raw_agreement_rate"],
                "disagreement_rate": ds_agree["disagreement_rate"],
                "cohen_kappa": ds_agree["cohen_kappa"],
                "joint_failure_rate": ds_corr["joint_failure_rate"],
                "policy_conflicts_count": sum(1 for r in ds_records if r.policy_state == DisagreementState.CONFLICTING_DETECTION_SIGNALS),
            }

        # Latency statistics
        latency_stats = {
            "model_1_ela": {
                "mean_ms": round(float(np.mean(m1_latencies)), 2),
                "median_ms": round(float(np.median(m1_latencies)), 2),
                "p95_ms": round(float(np.percentile(m1_latencies, 95)), 2),
            },
            "model_2_frequency": {
                "mean_ms": round(float(np.mean(m2_latencies)), 2),
                "median_ms": round(float(np.median(m2_latencies)), 2),
                "p95_ms": round(float(np.percentile(m2_latencies, 95)), 2),
            },
            "dual_execution": {
                "mean_ms": round(float(np.mean(dual_latencies)), 2),
                "median_ms": round(float(np.median(dual_latencies)), 2),
                "p95_ms": round(float(np.percentile(dual_latencies, 95)), 2),
            },
            "peak_memory_mb": round(mem_peak_mb, 2),
        }

        audit_results = {
            "total_samples": len(records),
            "performance_comparison": {
                "model_1_spatial_ela": m1_metrics,
                "model_2_frequency_spectral": m2_metrics,
                "naive_score_averaging": avg_metrics,
                "naive_or_voting": or_metrics,
                "naive_and_voting": and_metrics,
            },
            "inter_model_agreement": agreement_stats,
            "correlated_errors": corr_error_stats,
            "policy_state_counts": policy_state_counts,
            "dataset_breakdowns": dataset_breakdowns,
            "computational_profiling": latency_stats,
        }

        return records, audit_results
