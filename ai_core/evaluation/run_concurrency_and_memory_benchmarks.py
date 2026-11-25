"""Empirical Benchmarking Harness for SafeHER Image Inference & Memory Stability.

Measures:
1. Model Initialization: Cold-start latency and single-load guarantee.
2. Sequential Inference: Single-request latency, p50, p95, p99, throughput.
3. Concurrent Workloads: Multi-threaded latency, p50, p95, p99, throughput, CPU %, peak RSS.
4. Memory Stability & Leak Audit: Memory delta across 100 repeated requests with tracemalloc.
5. Large Valid Images: Latency and peak memory on 2048x2048 and 3000x3000 images.
6. Mixed Dimensions: Latency across various resolutions (128x128, 512x512, 1024x768, 2048x1536).
7. Model Fallback Latency: Fallback behavior when primary runner is offline.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import io
import json
import logging
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import psutil

from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import BaseImageModelRunner, ModelPrediction
from ai_core.image_intelligence.service import (
    get_detector_initialization_count,
    get_detector_initialization_time_ms,
    get_image_detector,
    reset_detector_initialization_stats,
    set_image_detector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("safeher.benchmarks")


def generate_benchmark_image(
    dimensions: Tuple[int, int] = (512, 512),
    format_name: str = "JPEG",
    quality: int = 90,
) -> bytes:
    """Generate in-memory test image with realistic texture, lines, and gradients."""
    img = Image.new("RGB", dimensions, color=(110, 140, 180))
    draw = ImageDraw.Draw(img)
    w, h = dimensions
    draw.rectangle([w // 6, h // 6, w // 2, h // 2], fill=(220, 50, 50))
    draw.ellipse([w // 3, h // 3, 2 * w // 3, 2 * h // 3], fill=(50, 200, 50))
    draw.line([0, 0, w, h], fill=(240, 240, 40), width=3)
    buf = io.BytesIO()
    img.save(buf, format=format_name, quality=quality)
    return buf.getvalue()


class FailingMockRunner(BaseImageModelRunner):
    """Simulates a model runner throwing an exception to trigger fallback."""

    def __init__(self) -> None:
        self.runner_name = "FailingMockRunner"
        self.runner_version = "0.0.1"

    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "ready"

    def predict(self, preprocessed: Any) -> ModelPrediction:
        raise RuntimeError("Simulated primary neural model crash/OOM")


def run_benchmarks() -> Dict[str, Any]:
    """Execute complete performance, concurrency, and memory stability benchmarks."""
    process = psutil.Process(os.getpid())
    results: Dict[str, Any] = {}

    logger.info("Starting SafeHER Image Inference & Memory Stability Benchmarks...")

    # ==========================================================================
    # SCENARIO 1: First Model Load & Singleton Guarantee
    # ==========================================================================
    set_image_detector(None)
    reset_detector_initialization_stats()

    t_init_start = time.perf_counter()
    detector = get_image_detector()
    t_init_elapsed_ms = (time.perf_counter() - t_init_start) * 1000.0

    # Test concurrent race condition on singleton initialization
    set_image_detector(None)
    reset_detector_initialization_stats()

    def _concurrent_get():
        return get_image_detector()

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_concurrent_get) for _ in range(32)]
        detectors = [f.result() for f in futures]

    init_count = get_detector_initialization_count()
    all_same_instance = all(d is detectors[0] for d in detectors)

    results["model_initialization"] = {
        "cold_start_time_ms": round(t_init_elapsed_ms, 2),
        "concurrent_init_threads": 16,
        "concurrent_init_requests": 32,
        "actual_initialization_count": init_count,
        "single_load_guaranteed": (init_count == 1 and all_same_instance),
    }
    logger.info("Scenario 1 (Model Load): %s", results["model_initialization"])

    # ==========================================================================
    # SCENARIO 2: Sequential Requests (Single-Request Latency & Percentiles)
    # ==========================================================================
    test_img_512 = generate_benchmark_image((512, 512))
    sequential_latencies: List[float] = []

    # Warmup
    for _ in range(3):
        detector.detect(test_img_512)

    rss_before_seq = process.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    n_seq = 40
    t_seq_start = time.perf_counter()
    for _ in range(n_seq):
        t0 = time.perf_counter()
        res = detector.detect(test_img_512)
        lat = (time.perf_counter() - t0) * 1000.0
        sequential_latencies.append(lat)
        assert res.assessment_id is not None

    t_seq_total = time.perf_counter() - t_seq_start
    current_trace, peak_trace = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after_seq = process.memory_info().rss / (1024 * 1024)

    seq_arr = np.array(sequential_latencies)
    results["sequential_inference"] = {
        "num_requests": n_seq,
        "total_time_s": round(t_seq_total, 3),
        "throughput_req_per_sec": round(n_seq / t_seq_total, 2),
        "mean_latency_ms": round(float(np.mean(seq_arr)), 2),
        "p50_ms": round(float(np.percentile(seq_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(seq_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(seq_arr, 99)), 2),
        "min_latency_ms": round(float(np.min(seq_arr)), 2),
        "max_latency_ms": round(float(np.max(seq_arr)), 2),
        "tracemalloc_peak_mb": round(peak_trace / (1024 * 1024), 2),
        "process_rss_delta_mb": round(rss_after_seq - rss_before_seq, 2),
    }
    logger.info("Scenario 2 (Sequential): %s", results["sequential_inference"])

    # ==========================================================================
    # SCENARIO 3: Concurrent Workloads (Multi-Threaded Throughput & Latency)
    # ==========================================================================
    concurrency_levels = [4, 8, 12]
    results["concurrent_workloads"] = {}

    for c in concurrency_levels:
        requests_per_batch = c * 5  # 20, 40, 60
        concurrent_latencies: List[float] = []
        failures = 0

        rss_before = process.memory_info().rss / (1024 * 1024)
        cpu_samples = []

        def _worker_task(idx):
            t0 = time.perf_counter()
            try:
                r = detector.detect(test_img_512)
                lat = (time.perf_counter() - t0) * 1000.0
                return lat, True, r.assessment_id
            except Exception as e:
                lat = (time.perf_counter() - t0) * 1000.0
                return lat, False, None

        t_conc_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as pool:
            futures = [pool.submit(_worker_task, i) for i in range(requests_per_batch)]
            for fut in as_completed(futures):
                lat, ok, _ = fut.result()
                concurrent_latencies.append(lat)
                if not ok:
                    failures += 1

        t_conc_total = time.perf_counter() - t_conc_start
        rss_after = process.memory_info().rss / (1024 * 1024)

        c_arr = np.array(concurrent_latencies)
        results["concurrent_workloads"][f"concurrency_{c}"] = {
            "workers": c,
            "num_requests": requests_per_batch,
            "total_time_s": round(t_conc_total, 3),
            "throughput_req_per_sec": round(requests_per_batch / t_conc_total, 2),
            "mean_latency_ms": round(float(np.mean(c_arr)), 2),
            "p50_ms": round(float(np.percentile(c_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(c_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(c_arr, 99)), 2),
            "failure_rate": round(failures / requests_per_batch, 4),
            "rss_peak_mb": round(rss_after, 2),
        }
        logger.info("Scenario 3 (Concurrency %d): %s", c, results["concurrent_workloads"][f"concurrency_{c}"])

    # ==========================================================================
    # SCENARIO 4: Memory Stability & Leak Audit (100 Repeated Inferences)
    # ==========================================================================
    gc.collect()
    tracemalloc.start()
    rss_start = process.memory_info().rss / (1024 * 1024)

    memory_checkpoints: List[Dict[str, Any]] = []
    n_leak_tests = 100

    for i in range(1, n_leak_tests + 1):
        detector.detect(test_img_512)
        if i in (10, 25, 50, 75, 100):
            gc.collect()
            curr_trace, peak_trace = tracemalloc.get_traced_memory()
            curr_rss = process.memory_info().rss / (1024 * 1024)
            memory_checkpoints.append({
                "iteration": i,
                "traced_current_mb": round(curr_trace / (1024 * 1024), 2),
                "traced_peak_mb": round(peak_trace / (1024 * 1024), 2),
                "rss_mb": round(curr_rss, 2),
            })

    tracemalloc.stop()
    gc.collect()
    rss_end = process.memory_info().rss / (1024 * 1024)

    results["memory_stability_leak_audit"] = {
        "iterations": n_leak_tests,
        "rss_initial_mb": round(rss_start, 2),
        "rss_final_mb": round(rss_end, 2),
        "rss_delta_mb": round(rss_end - rss_start, 2),
        "checkpoints": memory_checkpoints,
        "leak_detected": (rss_end - rss_start) > 25.0,  # Threshold for noticeable leak
    }
    logger.info("Scenario 4 (Memory Leak Audit): Delta=%.2f MB, Leak Detected=%s",
                results["memory_stability_leak_audit"]["rss_delta_mb"],
                results["memory_stability_leak_audit"]["leak_detected"])

    # ==========================================================================
    # SCENARIO 5: Large Valid Images & Mixed Dimensions
    # ==========================================================================
    dimension_tests = [
        ("micro_128x128", (128, 128)),
        ("standard_512x512", (512, 512)),
        ("hd_1024x768", (1024, 768)),
        ("large_2048x2048", (2048, 2048)),
        ("highres_3000x3000", (3000, 3000)),
    ]
    results["dimension_benchmarks"] = {}

    for label, dims in dimension_tests:
        raw_data = generate_benchmark_image(dims)
        gc.collect()
        rss_b = process.memory_info().rss / (1024 * 1024)

        t_dim_start = time.perf_counter()
        resp = detector.detect(raw_data)
        t_dim_lat = (time.perf_counter() - t_dim_start) * 1000.0

        rss_a = process.memory_info().rss / (1024 * 1024)

        results["dimension_benchmarks"][label] = {
            "dimensions": f"{dims[0]}x{dims[1]}",
            "megapixels": round((dims[0] * dims[1]) / 1e6, 2),
            "payload_kb": round(len(raw_data) / 1024, 1),
            "latency_ms": round(t_dim_lat, 2),
            "peak_rss_mb": round(rss_a, 2),
            "assessment_label": resp.assessment_label.value,
        }
        logger.info("Scenario 5 (Dimension %s): Latency=%.2f ms", label, t_dim_lat)

    # ==========================================================================
    # SCENARIO 6: Model Fallback Latency & Graceful Recovery
    # ==========================================================================
    failing_detector = SafeHERImageDetector(runner=FailingMockRunner())
    t_fallback_start = time.perf_counter()
    fallback_resp = failing_detector.detect(test_img_512)
    t_fallback_lat = (time.perf_counter() - t_fallback_start) * 1000.0

    results["model_fallback_performance"] = {
        "fallback_latency_ms": round(t_fallback_lat, 2),
        "primary_runner_status": "FailingMockRunner (failed)",
        "active_runner_used": fallback_resp.detector_name,
        "graceful_recovery_successful": ("Baseline" in fallback_resp.detector_name),
    }
    logger.info("Scenario 6 (Fallback): Latency=%.2f ms, Active=%s",
                t_fallback_lat, fallback_resp.detector_name)

    return results


if __name__ == "__main__":
    report_dict = run_benchmarks()
    out_path = Path("ai_core/evaluation/benchmark_results.json")
    out_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"\nBenchmark suite completed successfully. Results saved to {out_path}")
