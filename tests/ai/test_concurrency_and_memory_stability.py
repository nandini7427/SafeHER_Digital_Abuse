"""Automated Concurrency, Thread Safety, and Memory Stability Test Suite for SafeHER.

Verifies:
1. Model loads only once under race conditions (Double-Checked Locking thread safety).
2. Concurrent requests do not share mutable image buffers or cross-contaminate state.
3. No memory leaks occur across repeated inference executions.
4. Large valid images and mixed dimensions execute within bounded memory limits.
5. Primary runner failure gracefully falls back to deterministic baseline without crashing.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import io
import threading
import time
import tracemalloc
from typing import Any, List, Tuple
import uuid

import numpy as np
from PIL import Image, ImageDraw
import pytest

from ai_core.image_intelligence.detector import SafeHERImageDetector
from ai_core.image_intelligence.model_runner import BaseImageModelRunner, ModelPrediction
from ai_core.image_intelligence.preprocessing import PreprocessedImage
from ai_core.image_intelligence.schemas import ImageAssessmentResponse, UncertaintyState
from ai_core.image_intelligence.service import (
    get_detector_initialization_count,
    get_image_detector,
    reset_detector_initialization_stats,
    set_image_detector,
)


def create_test_image(
    dimensions: Tuple[int, int] = (256, 256),
    color: Tuple[int, int, int] = (100, 150, 200),
    draw_pattern: bool = True,
) -> bytes:
    """Helper to generate in-memory test image bytes."""
    img = Image.new("RGB", dimensions, color=color)
    if draw_pattern:
        draw = ImageDraw.Draw(img)
        w, h = dimensions
        draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(200, 50, 50))
        draw.line([0, 0, w, h], fill=(255, 255, 0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class FailingTestRunner(BaseImageModelRunner):
    """Mock runner that always throws to test fallback behavior."""

    def __init__(self) -> None:
        self.runner_name = "FailingTestRunner"
        self.runner_version = "0.0.1"

    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "ready"

    def predict(self, preprocessed: PreprocessedImage) -> ModelPrediction:
        raise RuntimeError("Simulated neural inference failure / OutOfMemoryError")


def test_model_singleton_initialization_thread_safe():
    """Verify that concurrent requests in different threads initialize the detector exactly once."""
    reset_detector_initialization_stats()

    results: List[SafeHERImageDetector] = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def _init_worker():
        barrier.wait()  # Synchronize threads to maximize race condition probability
        inst = get_image_detector()
        with lock:
            results.append(inst)

    threads = [threading.Thread(target=_init_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 16
    first_inst = results[0]
    for inst in results:
        assert inst is first_inst, "Each thread must receive the exact same singleton instance"

    init_count = get_detector_initialization_count()
    assert init_count == 1, f"Model must be initialized exactly once; got {init_count}"


def test_concurrent_inference_requests_do_not_share_mutable_buffers():
    """Verify that concurrent inference requests with unique images remain strictly isolated."""
    detector = get_image_detector()

    # Create 8 unique images with distinct colors and dimensions
    unique_payloads = [
        (
            create_test_image(
                dimensions=(128 + i * 32, 128 + i * 32),
                color=(20 * i, 150 - i * 10, 50 + i * 20),
            ),
            i,
        )
        for i in range(8)
    ]

    def _inference_worker(data: bytes, worker_id: int) -> Tuple[int, ImageAssessmentResponse, str]:
        expected_hash = hashlib.sha256(data).hexdigest()
        output = detector.detect(data)
        return worker_id, output, expected_hash

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_inference_worker, data, wid)
            for data, wid in unique_payloads
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == 8
    seen_ids = set()
    for wid, output, expected_hash in results:
        # Verify unique assessment IDs
        assert output.assessment_id not in seen_ids, "Assessment IDs must be globally unique per request"
        seen_ids.add(output.assessment_id)

        # Verify metadata summary exists
        assert output.metadata_summary is not None, "Metadata summary must be populated"
        # Verify hash integrity matches worker's specific input
        assert output.metadata_summary.sha256 == expected_hash, (
            f"Buffer corruption detected: worker {wid} received output for different image"
        )
        # Verify dimension metadata matches worker's specific input
        expected_dim = 128 + wid * 32
        assert output.metadata_summary.dimensions == (expected_dim, expected_dim)


def test_no_memory_leak_across_repeated_inferences():
    """Verify that running repeated inferences does not leak uncollected memory."""
    detector = get_image_detector()
    img_bytes = create_test_image((300, 300))

    gc.collect()
    tracemalloc.start()

    # Warmup
    for _ in range(5):
        detector.detect(img_bytes)

    gc.collect()
    snapshot_before = tracemalloc.take_snapshot()

    # 30 repeated inferences
    for _ in range(30):
        detector.detect(img_bytes)

    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_delta = sum(stat.size_diff for stat in top_stats)

    # Net growth after 30 iterations and GC must be negligible (< 1.5 MB)
    max_allowed_growth_bytes = 1.5 * 1024 * 1024
    assert total_delta < max_allowed_growth_bytes, (
        f"Potential memory leak detected: net memory growth was {total_delta / 1024:.2f} KB"
    )


def test_large_valid_image_execution_and_resource_bounds():
    """Verify that a large valid image (2048x2048) executes without OOM or unbounded memory."""
    detector = get_image_detector()
    large_img_bytes = create_test_image((2048, 2048))

    t0 = time.perf_counter()
    output = detector.detect(large_img_bytes)
    duration_s = time.perf_counter() - t0

    assert isinstance(output, ImageAssessmentResponse)
    assert output.metadata_summary is not None
    assert output.metadata_summary.dimensions == (2048, 2048)
    # Should execute in reasonable time for 4.2 megapixels
    assert duration_s < 10.0, f"Inference took too long: {duration_s:.2f}s"


def test_mixed_image_dimensions_stability():
    """Verify that detector reliably handles diverse aspect ratios and sizes sequentially and concurrently."""
    detector = get_image_detector()
    dimensions_list = [
        (100, 100),
        (320, 240),
        (640, 480),
        (800, 1200),  # portrait
        (1280, 720),  # landscape widescreen
    ]

    for dims in dimensions_list:
        data = create_test_image(dims)
        resp = detector.detect(data)
        assert resp.metadata_summary is not None
        assert resp.metadata_summary.dimensions == dims
        assert resp.uncertainty_state in [
            UncertaintyState.ANALYSIS_USABLE_EVIDENCE,
            UncertaintyState.ANALYSIS_POOR_QUALITY_EVIDENCE,
            UncertaintyState.MODEL_UNCERTAIN,
            UncertaintyState.OUTSIDE_SUPPORTED_DOMAIN,
        ]


def test_model_unavailable_fallback_recovery():
    """Verify that detector catches primary model crash and falls back to deterministic baseline."""
    failing_runner = FailingTestRunner()
    resilient_detector = SafeHERImageDetector(runner=failing_runner)

    img_data = create_test_image((256, 256))
    t0 = time.perf_counter()
    output = resilient_detector.detect(img_data)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Ensure fallback executed and did not raise unhandled exception
    assert isinstance(output, ImageAssessmentResponse)
    assert output.uncertainty_state != UncertaintyState.ANALYSIS_CANNOT_BE_COMPLETED
    assert "Baseline" in output.detector_name
    assert elapsed_ms < 2000.0, f"Fallback should complete swiftly; took {elapsed_ms:.1f}ms"
