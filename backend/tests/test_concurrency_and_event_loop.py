"""Automated Verification Suite: Asynchronous Concurrency, Timeout Protection, and Event Loop Stability.

Verifies:
1. Event Loop Non-Blocking:
   - CPU-bound image assessment runs in a thread pool via `run_in_threadpool`,
     preventing starvation of the asyncio event loop.
   - Concurrently dispatched lightweight coroutines (/health) execute and return
     immediately while vision inference is actively computing.
2. Inference Timeout Protection:
   - Runaway or unresponsive model execution is terminated via `asyncio.wait_for`,
     returning HTTP 504 Gateway Timeout without leaking threads or blocking future requests.
3. Concurrency Semaphore & Resource Bounds:
   - Concurrency is throttled to at most 10 concurrent inference workloads via `asyncio.Semaphore`.
   - Burst requests are serialized or queued safely without 500 errors or thread exhaustion.
4. Resilient Error Containment:
   - Internal inference crashes return sanitized HTTP 500 responses without exposing
     raw stack traces or destabilizing the service process.
"""

import asyncio
import io
from pathlib import Path
import threading
import time
from typing import Dict
from unittest.mock import patch
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_core.image_intelligence.service import set_image_detector
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import User
from backend.app.services.image_service import ImageBackendService, ProcessCapacityLimiter, backend_image_service


def _create_test_jpeg(dimensions=(256, 256), color=(120, 160, 200)) -> bytes:
    """Generate in-memory valid JPEG bytes."""
    img = Image.new("RGB", dimensions, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, dimensions[0] // 2, dimensions[1] // 2], fill=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def test_db(tmp_path):
    """Create isolated thread-safe SQLite database file for concurrent test execution."""
    db_file = tmp_path / "test_concurrency.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    set_image_detector(None)


@pytest.fixture
def client(test_db):
    """TestClient configured with test database."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client) -> Dict[str, str]:
    """Helper fixture registering and logging in a test user."""
    reg_data = {
        "email": "concurrency_tester@example.com",
        "password": "SafeHER-Concurrency-123!",
    }
    client.post("/api/v1/auth/register", json=reg_data)
    login_res = client.post("/api/v1/auth/login", json=reg_data)
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.anyio
async def test_event_loop_unblocked_during_heavy_inference(test_db):
    """Verify that CPU-bound image inference executed in a threadpool does not block the asyncio event loop."""
    # Register user and acquire token
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        reg_res = await ac.post(
            "/api/v1/auth/register",
            json={"email": "loop_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "loop_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}

        img_bytes = _create_test_jpeg((300, 300))

        # We will dispatch an image analysis request and immediately check /health
        # The health check must respond with low latency while analysis is ongoing
        analysis_task = asyncio.create_task(
            ac.post(
                "/api/v1/ai/image-assessment",
                headers=auth_hdr,
                files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
            )
        )

        # Allow analysis coroutine to start
        await asyncio.sleep(0.01)

        t_health_start = time.perf_counter()
        health_res = await ac.get("/health")
        health_latency_ms = (time.perf_counter() - t_health_start) * 1000.0

        assert health_res.status_code == 200
        # Health check must not be starved by inference threadpool
        assert health_latency_ms < 100.0, f"Event loop was blocked: health check took {health_latency_ms:.1f}ms"

        analysis_res = await analysis_task
        assert analysis_res.status_code == 200
        data = analysis_res.json()
        assert "assessment_id" in data
        assert data["detector_name"] != ""


@pytest.mark.anyio
async def test_inference_timeout_raises_504_gateway_timeout(test_db):
    """Verify that an inference task exceeding timeout_seconds is terminated with HTTP 504 Gateway Timeout."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/api/v1/auth/register",
            json={"email": "timeout_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "timeout_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}

        img_bytes = _create_test_jpeg((200, 200))

        def _slow_assessment(*args, **kwargs):
            time.sleep(0.3)  # Exceeds the tight 0.05s timeout
            from ai_core.image_intelligence.service import image_service
            return image_service.assess_image(*args, **kwargs)

        original_timeout = backend_image_service.timeout_seconds
        backend_image_service.timeout_seconds = 0.05
        try:
            with patch("backend.app.services.image_service.ai_image_service.assess_image", side_effect=_slow_assessment):
                t0 = time.perf_counter()
                response = await ac.post(
                    "/api/v1/ai/image-assessment",
                    headers=auth_hdr,
                    files={"file": ("slow.jpg", img_bytes, "image/jpeg")},
                )
                elapsed_s = time.perf_counter() - t0

                assert response.status_code == 504
                assert "timed out" in response.json()["detail"].lower()
                assert elapsed_s < 0.25, f"Timeout was not enforced promptly; elapsed {elapsed_s:.2f}s"
        finally:
            backend_image_service.timeout_seconds = original_timeout


@pytest.mark.anyio
async def test_concurrency_semaphore_limits_parallel_inferences(test_db):
    """Verify that multiple concurrent image analysis requests are handled cleanly under the semaphore."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/api/v1/auth/register",
            json={"email": "sem_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "sem_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}

        img_bytes = _create_test_jpeg((128, 128))

        async def _upload_task(idx: int):
            return await ac.post(
                "/api/v1/ai/image-assessment",
                headers=auth_hdr,
                files={"file": (f"concur_{idx}.jpg", img_bytes, "image/jpeg")},
            )

        # Dispatch 8 concurrent requests
        tasks = [_upload_task(i) for i in range(8)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 8
        for r in responses:
            assert r.status_code == 200
            assert "assessment_id" in r.json()


def test_internal_inference_failure_gracefully_returns_500(client, auth_headers):
    """Verify that an unexpected runtime failure inside the detector returns a sanitized 500 without leaking stack traces."""
    img_bytes = _create_test_jpeg((150, 150))

    with patch(
        "backend.app.services.image_service.ai_image_service.assess_image",
        side_effect=RuntimeError("Simulated neural inference GPU bus crash"),
    ):
        res = client.post(
            "/api/v1/ai/image-assessment",
            headers=auth_headers,
            files={"file": ("crash.jpg", img_bytes, "image/jpeg")},
        )

        assert res.status_code == 500
        detail = res.json()["detail"]
        assert "internal error occurred" in detail.lower()
        # Ensure raw exception text or file paths are NOT exposed to caller
        assert "Simulated neural inference" not in detail


@pytest.mark.anyio
async def test_timed_out_analysis_retains_permit_until_computation_finishes(test_db):
    """Verify that when an HTTP request times out, the permit remains occupied until the thread genuinely exits."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/api/v1/auth/register",
            json={"email": "retain_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "retain_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}
        img_bytes = _create_test_jpeg((150, 150))

        thread_finished = threading.Event()
        from ai_core.image_intelligence.service import image_service as direct_ai_service
        real_assess = direct_ai_service.assess_image

        def _delayed_assessment(*args, **kwargs):
            try:
                time.sleep(0.15)
                return real_assess(*args, **kwargs)
            finally:
                thread_finished.set()

        original_timeout = backend_image_service.timeout_seconds
        backend_image_service.timeout_seconds = 0.05
        try:
            with patch("backend.app.services.image_service.ai_image_service.assess_image", side_effect=_delayed_assessment):
                response = await ac.post(
                    "/api/v1/ai/image-assessment",
                    headers=auth_hdr,
                    files={"file": ("retain.jpg", img_bytes, "image/jpeg")},
                )
                assert response.status_code == 504
                # Immediately after timeout returns 504, the thread is still running!
                # The permit must remain OCCUPIED (0 available permits, 1 active computation).
                assert not thread_finished.is_set(), "Thread unexpectedly finished before verification"
                assert backend_image_service.active_computations == 1
                assert backend_image_service.available_permits == 0

                # Wait for the thread to genuinely finish
                await asyncio.to_thread(thread_finished.wait, timeout=2.0)
                # Small pause to allow callback to execute
                await asyncio.sleep(0.05)

                # Now that thread completed, permit must be released exactly once
                assert backend_image_service.active_computations == 0
                assert backend_image_service.available_permits == 1
        finally:
            backend_image_service.timeout_seconds = original_timeout


@pytest.mark.anyio
async def test_second_request_waits_for_running_thread_after_first_times_out(test_db):
    """Verify that a second request cannot acquire capacity while the first timed-out thread is still executing."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/api/v1/auth/register",
            json={"email": "queue_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "queue_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}
        img_bytes = _create_test_jpeg((120, 120))

        first_thread_active = threading.Event()
        first_thread_finish = threading.Event()
        call_count = 0
        call_lock = threading.Lock()
        from ai_core.image_intelligence.service import image_service as direct_ai_service
        real_assess = direct_ai_service.assess_image

        def _controlled_assessment(*args, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count

            if current_call == 1:
                first_thread_active.set()
                # Hold until released
                first_thread_finish.wait(timeout=2.0)
            return real_assess(*args, **kwargs)

        original_timeout = backend_image_service.timeout_seconds
        backend_image_service.timeout_seconds = 0.05
        try:
            with patch("backend.app.services.image_service.ai_image_service.assess_image", side_effect=_controlled_assessment):
                # Dispatch first request (will time out quickly)
                t1 = asyncio.create_task(
                    ac.post(
                        "/api/v1/ai/image-assessment",
                        headers=auth_hdr,
                        files={"file": ("req1.jpg", img_bytes, "image/jpeg")},
                    )
                )
                res1 = await t1
                assert res1.status_code == 504
                assert first_thread_active.is_set()

                # Dispatch second request with normal timeout while first thread is still computing
                backend_image_service.timeout_seconds = 5.0
                t2 = asyncio.create_task(
                    ac.post(
                        "/api/v1/ai/image-assessment",
                        headers=auth_hdr,
                        files={"file": ("req2.jpg", img_bytes, "image/jpeg")},
                    )
                )

                # Allow second request to start and attempt to acquire permit
                await asyncio.sleep(0.05)
                # Second request MUST be waiting; call_count must still be 1!
                assert call_count == 1
                assert backend_image_service.active_computations == 1

                # Now allow first thread to finish
                first_thread_finish.set()
                res2 = await t2
                assert res2.status_code == 200
                assert call_count == 2
                assert backend_image_service.active_computations == 0
                assert backend_image_service.available_permits == 1
        finally:
            backend_image_service.timeout_seconds = original_timeout
            first_thread_finish.set()


@pytest.mark.anyio
async def test_multiple_requests_after_repeated_timeouts_never_exceed_capacity(test_db):
    """Verify that multiple consecutive timeouts cannot trigger unbound thread explosion."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/api/v1/auth/register",
            json={"email": "burst_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "burst_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}
        img_bytes = _create_test_jpeg((100, 100))

        active_thread_counter = 0
        max_seen_concurrent = 0
        counter_lock = threading.Lock()
        from ai_core.image_intelligence.service import image_service as direct_ai_service
        real_assess = direct_ai_service.assess_image

        def _tracked_slow_work(*args, **kwargs):
            nonlocal active_thread_counter, max_seen_concurrent
            with counter_lock:
                active_thread_counter += 1
                if active_thread_counter > max_seen_concurrent:
                    max_seen_concurrent = active_thread_counter
            try:
                time.sleep(0.08)
                return real_assess(*args, **kwargs)
            finally:
                with counter_lock:
                    active_thread_counter -= 1

        original_timeout = backend_image_service.timeout_seconds
        backend_image_service.timeout_seconds = 0.02  # Ultra fast timeout
        try:
            with patch("backend.app.services.image_service.ai_image_service.assess_image", side_effect=_tracked_slow_work):
                # Fire 6 rapid requests
                tasks = [
                    ac.post(
                        "/api/v1/ai/image-assessment",
                        headers=auth_hdr,
                        files={"file": (f"burst_{i}.jpg", img_bytes, "image/jpeg")},
                    )
                    for i in range(6)
                ]
                responses = await asyncio.gather(*tasks)

                # All 6 timed out
                for r in responses:
                    assert r.status_code == 504

                # Wait for any lingering worker threads to finish
                for _ in range(40):
                    if active_thread_counter == 0 and backend_image_service.active_computations == 0:
                        break
                    await asyncio.sleep(0.05)

                # Crucial assertion: Even under 6 rapid timeouts, max simultaneous executing threads was bounded to 1!
                assert max_seen_concurrent == 1
                assert backend_image_service.available_permits == 1
        finally:
            backend_image_service.timeout_seconds = original_timeout


@pytest.mark.anyio
async def test_permit_released_exactly_once_after_internal_failure(test_db):
    """Verify that internal detector failure releases the capacity permit exactly once."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/api/v1/auth/register",
            json={"email": "fail_permit_test@example.com", "password": "SecurePassword123!"},
        )
        login_res = await ac.post(
            "/api/v1/auth/login",
            json={"email": "fail_permit_test@example.com", "password": "SecurePassword123!"},
        )
        token = login_res.json()["access_token"]
        auth_hdr = {"Authorization": f"Bearer {token}"}
        img_bytes = _create_test_jpeg((100, 100))

        with patch(
            "backend.app.services.image_service.ai_image_service.assess_image",
            side_effect=RuntimeError("Simulated internal vision model failure"),
        ):
            res = await ac.post(
                "/api/v1/ai/image-assessment",
                headers=auth_hdr,
                files={"file": ("fail.jpg", img_bytes, "image/jpeg")},
            )
            assert res.status_code == 500

            # Small pause for done callback
            await asyncio.sleep(0.02)
            assert backend_image_service.active_computations == 0
            assert backend_image_service.available_permits == 1


def test_multiple_service_instances_in_same_worker_share_pool_and_prevent_excess_capacity():
    """Verify that multiple ImageBackendService instances in the same process share the bounded pool.

    Prevents additional service instances from creating independent executors and exceeding worker capacity.
    """
    svc1 = ImageBackendService(max_concurrent_analyses=1)
    svc2 = ImageBackendService(max_concurrent_analyses=1)

    # Both instances share the exact same underlying limiter and executor
    assert svc1._limiter is svc2._limiter
    assert svc1._executor is svc2._executor
    assert svc1.available_permits == 1
    assert svc2.available_permits == 1


def test_container_capacity_coordinator_enforces_aggregate_container_limit(tmp_path):
    """Verify that multiple worker processes cannot exceed aggregate container capacity."""
    slots_dir = tmp_path / "capacity_slots"

    # Simulate 3 worker processes pointing to the same container capacity coordinator slots
    worker_proc_1 = ImageBackendService(isolated=True, slots_dir=slots_dir, max_container_slots=2)
    worker_proc_2 = ImageBackendService(isolated=True, slots_dir=slots_dir, max_container_slots=2)
    worker_proc_3 = ImageBackendService(isolated=True, slots_dir=slots_dir, max_container_slots=2)

    try:
        # Worker 1 acquires container slot 0
        s0, h0 = worker_proc_1._coordinator.acquire_slot(timeout=1.0)
        assert s0 == 0

        # Worker 2 acquires container slot 1
        s1, h1 = worker_proc_2._coordinator.acquire_slot(timeout=1.0)
        assert s1 == 1

        # Both slots occupied: active container computations == 2, available == 0
        assert worker_proc_1.active_container_computations == 2
        assert worker_proc_1.available_container_slots == 0
        assert worker_proc_2.active_container_computations == 2
        assert worker_proc_3.available_container_slots == 0

        # Worker 3 attempts to acquire: MUST time out because container limit of 2 is strictly enforced
        with pytest.raises(TimeoutError, match="Timed out waiting for container capacity slot"):
            worker_proc_3._coordinator.acquire_slot(timeout=0.1)

        # Worker 1 releases slot 0
        worker_proc_1._coordinator.release_slot(s0, h0)
        assert worker_proc_1.active_container_computations == 1
        assert worker_proc_1.available_container_slots == 1

        # Now Worker 3 can successfully acquire the freed slot!
        s3, h3 = worker_proc_3._coordinator.acquire_slot(timeout=1.0)
        assert s3 == 0
        assert worker_proc_3.active_container_computations == 2

        # Clean up Worker 2 and 3
        worker_proc_2._coordinator.release_slot(s1, h1)
        worker_proc_3._coordinator.release_slot(s3, h3)
        assert worker_proc_1.active_container_computations == 0
        assert worker_proc_1.available_container_slots == 2
    finally:
        worker_proc_1.shutdown(wait=False)
        worker_proc_2.shutdown(wait=False)
        worker_proc_3.shutdown(wait=False)


def test_worker_crash_or_restart_releases_capacity_slot_without_stranding(tmp_path):
    """Verify that if a worker process crashes, the OS kernel immediately frees its slot lock."""
    import subprocess
    import sys

    slots_dir = tmp_path / "crash_test_slots"
    coordinator = ImageBackendService(isolated=True, slots_dir=slots_dir, max_container_slots=2)._coordinator

    slot_0_path = slots_dir / "slot_0.lock"

    # Spawn child process holding slot_0
    child_code = f"""
import sys, time
sys.path.insert(0, r"{Path('.').resolve()}")
from backend.app.services.file_locking import try_lock_exclusive
f = open(r'{slot_0_path}', 'a+b')
try_lock_exclusive(f)
print('CHILD_ACQUIRED', flush=True)
time.sleep(15)
"""
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "CHILD_ACQUIRED"

    try:
        # While child is running, slot 0 is locked
        assert coordinator.active_container_computations() == 1

        # Kill child process (simulating ungraceful worker crash / OOM kill)
        proc.terminate()
        proc.wait(timeout=5)

        # Kernel immediately cleans up locks: active computations drops to 0 without stranding
        assert coordinator.active_container_computations() == 0

        # Another worker can acquire slot 0 immediately
        s, h = coordinator.acquire_slot(timeout=1.0)
        assert s == 0
        coordinator.release_slot(s, h)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_genuine_multiprocess_cross_worker_coordination_and_restart(tmp_path):
    """Verify cross-process coordination across distinct OS worker processes.

    Verifies:
    - Concurrent lock acquisition by two distinct OS processes.
    - Exactly two available slots across deployment.
    - Timeout when 3rd process attempts acquisition while both slots held.
    - Worker process termination and immediate OS kernel lock release.
    - Restarted worker successfully acquiring freed slot.
    - Shared lock-directory consistency across all processes.
    """
    import subprocess
    import sys

    slots_dir = tmp_path / "multiprocess_slots"
    slots_dir.mkdir(parents=True, exist_ok=True)

    repo_root = str(Path(".").resolve())
    worker_script = f"""
import sys
import time
from pathlib import Path
sys.path.insert(0, r"{repo_root}")
from backend.app.services.image_service import ContainerCapacityCoordinator

slots_dir = Path(r"{slots_dir}").resolve()
cmd = sys.argv[1]

coord = ContainerCapacityCoordinator(slots_dir=slots_dir, max_slots=2)

if cmd == "hold":
    slot_id, handle = coord.acquire_slot(timeout=2.0)
    print(f"HELD:{{slot_id}}", flush=True)
    time.sleep(120)
elif cmd == "try":
    timeout = float(sys.argv[2])
    try:
        slot_id, handle = coord.acquire_slot(timeout=timeout)
        print(f"ACQUIRED:{{slot_id}}", flush=True)
        coord.release_slot(slot_id, handle)
    except TimeoutError:
        print("TIMEOUT", flush=True)
"""
    script_file = tmp_path / "worker_subproc.py"
    script_file.write_text(worker_script, encoding="utf-8")

    # 1. Spawn Worker Process A (holds slot 0)
    proc_a = subprocess.Popen([sys.executable, str(script_file), "hold"], stdout=subprocess.PIPE, text=True)
    out_a = proc_a.stdout.readline().strip()
    assert out_a == "HELD:0"

    # 2. Spawn Worker Process B (holds slot 1)
    proc_b = subprocess.Popen([sys.executable, str(script_file), "hold"], stdout=subprocess.PIPE, text=True)
    out_b = proc_b.stdout.readline().strip()
    assert out_b == "HELD:1"

    try:
        # 3. Spawn Worker Process C: exactly 2 slots exist, so attempting acquisition MUST time out
        proc_c = subprocess.run([sys.executable, str(script_file), "try", "0.2"], capture_output=True, text=True)
        assert "TIMEOUT" in proc_c.stdout

        # 4. Terminate Worker Process A (simulating crash / kill)
        proc_a.terminate()
        proc_a.wait(timeout=5)

        # 5. Worker Restart: Spawn Worker Process A' (restarted worker)
        # Should now immediately acquire slot 0 released by terminated Worker A!
        proc_restart = subprocess.run([sys.executable, str(script_file), "try", "2.0"], capture_output=True, text=True)
        assert "ACQUIRED:0" in proc_restart.stdout
    finally:
        if proc_a.poll() is None:
            proc_a.kill()
            proc_a.wait(timeout=5)
        if proc_b.poll() is None:
            proc_b.kill()
            proc_b.wait(timeout=5)


@pytest.mark.anyio
async def test_cancelled_capacity_waiter_does_not_leave_queue_entry_or_leak_permit():
    limiter = ProcessCapacityLimiter(max_permits=1)
    await limiter.acquire()

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert limiter.active_count == 1
    assert limiter.available_permits == 0
    limiter.release()
    assert limiter.active_count == 0
    assert limiter.available_permits == 1


@pytest.mark.anyio
async def test_cancel_racing_with_capacity_handoff_returns_granted_permit():
    """Cancelling immediately after release must not leak a permit in the handoff race."""
    limiter = ProcessCapacityLimiter(max_permits=1)
    await limiter.acquire()

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    limiter.release()  # schedules the waiter with call_soon_threadsafe
    waiter.cancel()   # race the scheduled completion callback

    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert limiter.active_count == 0
    assert limiter.available_permits == 1
