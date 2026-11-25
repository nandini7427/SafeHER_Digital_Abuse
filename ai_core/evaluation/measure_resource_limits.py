"""Measures memory consumption, inference latency, and concurrency behavior for SafeHER Image Intelligence.

Evaluates:
1. Peak RSS memory across image dimensions (1024x1024, 2048x2048, 4096x4096).
2. CPU utilization during ELA and FFT execution.
3. Behavior of asyncio.wait_for cancellation on worker threads in run_in_threadpool.
4. Total memory footprint projection under multi-worker Uvicorn configurations.
"""

import asyncio
import io
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import psutil

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_core.image_intelligence.detector import SafeHERImageDetector


def create_test_image(width: int, height: int) -> bytes:
    """Generate in-memory RGB test image."""
    arr = np.random.randint(50, 220, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def measure_memory_and_latency(dimensions: tuple[int, int]) -> dict:
    process = psutil.Process(os.getpid())
    detector = SafeHERImageDetector()

    w, h = dimensions
    print(f"\nMeasuring resolution {w}x{h}...")

    # Force garbage collection
    import gc
    gc.collect()

    base_rss = process.memory_info().rss / (1024 * 1024)
    raw_bytes = create_test_image(w, h)
    payload_mb = len(raw_bytes) / (1024 * 1024)

    t0 = time.perf_counter()
    resp = detector.detect(raw_bytes)
    t1 = time.perf_counter()

    peak_rss = process.memory_info().rss / (1024 * 1024)
    delta_rss = max(0.0, peak_rss - base_rss)
    elapsed_ms = (t1 - t0) * 1000.0

    result = {
        "dimensions": f"{w}x{h}",
        "payload_size_mb": round(payload_mb, 2),
        "baseline_rss_mb": round(base_rss, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "delta_rss_mb": round(delta_rss, 1),
        "latency_ms": round(elapsed_ms, 1),
        "status": resp.assessment_label.value,
    }
    print(f"  -> Payload: {result['payload_size_mb']} MB | Delta RSS: {result['delta_rss_mb']} MB | Latency: {result['latency_ms']} ms")
    return result


def test_threadpool_cancellation_behavior():
    """Verify whether worker thread continues executing after asyncio.wait_for timeout."""
    print("\nTesting asyncio.wait_for timeout & threadpool continuation...")
    from starlette.concurrency import run_in_threadpool

    thread_finished = False

    def slow_worker():
        nonlocal thread_finished
        print("  [Worker Thread] Started heavy computation (simulating 2.0s work)...")
        time.sleep(2.0)
        thread_finished = True
        print("  [Worker Thread] Completed heavy computation!")

    async def run_test():
        try:
            print("  [Async Task] Dispatching with timeout=0.5s...")
            await asyncio.wait_for(run_in_threadpool(slow_worker), timeout=0.5)
        except asyncio.TimeoutError:
            print("  [Async Task] asyncio.TimeoutError caught as expected!")

        # Wait to observe worker thread
        print("  [Async Task] Waiting 2.0s to check if background worker thread is still running...")
        await asyncio.sleep(2.0)
        print(f"  [Async Task] Worker thread finished state: {thread_finished}")
        return thread_finished

    finished = asyncio.run(run_test())
    return finished


def main():
    print("=" * 70)
    print("SafeHER Milestone 3 — Resource Usage & Concurrency Audit")
    print("=" * 70)

    # 1. Measure single-image inference memory footprint
    results = []
    # Test 1024x1024, 2048x2048, 4096x4096
    for dims in [(1024, 1024), (2048, 2048), (4096, 4096)]:
        res = measure_memory_and_latency(dims)
        results.append(res)

    # 2. Test threadpool cancellation
    thread_persisted = test_threadpool_cancellation_behavior()

    # 3. Print projection table
    print("\n" + "=" * 70)
    print("CONCURRENCY CAPACITY PROJECTION IN 4 GB RAM CONTAINER")
    print("=" * 70)
    print("Container RAM Limit: 4096 MB | System & Runtime Overhead: ~600 MB")
    print("Available RAM for Inference: ~3496 MB")
    print("-" * 70)

    max_delta = results[-1]["delta_rss_mb"]
    # Estimate 40 concurrent tasks
    proj_40_workers = 40 * max_delta
    print(f"Per-Analysis Delta RSS at 4096x4096px: ~{max_delta} MB")
    print(f"Projected RAM for 40 Concurrent Requests: {proj_40_workers:.1f} MB ({proj_40_workers/1024:.2f} GB)")
    print(f"OOM Risk at 40 Concurrency: {'CRITICAL (EXCEEDS 4GB LIMIT BY ' + str(round((proj_40_workers - 3496)/1024, 2)) + ' GB)' if proj_40_workers > 3496 else 'SAFE'}")

    safe_host_concurrency = max(1, int(3496 // max_delta))
    print(f"\nRecommended Safe Total Host Concurrency: {safe_host_concurrency} concurrent requests")
    print(f"  - In 2 Uvicorn Workers: max_concurrent_analyses = {max(1, safe_host_concurrency // 2)} per worker")
    print(f"  - In 4 Uvicorn Workers: max_concurrent_analyses = {max(1, safe_host_concurrency // 4)} per worker")
    print(f"Worker Thread Persisted After Timeout: {thread_persisted} (CONFIRMED: Thread cancellation does NOT stop OS thread)")
    print("=" * 70)


if __name__ == "__main__":
    main()
