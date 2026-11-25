# SafeHER Milestone 3 — Code Audit & Correction Notes

## Scope

This package contains only the source files modified or added during the focused Milestone 3 code audit. It preserves the existing project structure and does not contain `.env`, `.git`, `private_evidence`, `node_modules`, virtual environments, cache files, or model weights.

The original uploaded ZIP was not modified.

## Corrections made

### Upload admission and request-body limits

- Added `backend/app/upload_middleware.py`, a pure ASGI middleware that runs before multipart parsing.
- Checks malformed and duplicate `Content-Length` values and rejects declared oversize bodies early.
- Counts actual received body bytes, so missing or understated `Content-Length` cannot bypass the request-body cap.
- Once the byte limit is crossed, returns `http.disconnect` for all subsequent reads so the multipart parser cannot consume more request-body chunks.
- Uses a deployment-shared upload-slot coordinator and releases the slot in a `finally` block.
- Added tests for missing/understated content length, over-limit bodies, admission exhaustion and not consuming chunks after rejection.

### Cross-process capacity and inference timeouts

- Added `backend/app/services/file_locking.py` using native OS advisory locks (`fcntl.flock` on POSIX and `msvcrt.locking` on Windows); no third-party file-lock dependency is required.
- Updated image-analysis and upload coordinators to use the native lock helpers.
- Reworked the process limiter's waiter lifecycle to handle cancellation racing with a permit handoff.
- Shielded the executor future from HTTP timeout/request cancellation so a running worker thread retains its capacity permit until the computation really finishes.
- The service now respects `WORKER_MAX_CONCURRENT_ANALYSES` when `IMAGE_CONCURRENCY_SEMAPHORE` is not explicitly supplied.
- Added genuine OS-subprocess coordination/restart coverage and cancellation/timeout tests.

### Evidence vault and cleanup lifecycle

- Fixed Windows-style path separators in user-supplied filenames before persisting filename metadata.
- Replaced string-prefix path containment checks with `Path.relative_to` validation.
- Corrected canonical `.part` to `.bin` lease-key mapping.
- Recovery now recognizes both `.bin` and matching `.part` files when examining stale cleanup leases.
- A stale cleanup claim that never reached a filesystem operation is released so a later scan can retry it.
- Quarantine moves are compensated back to the original path when a database status commit fails; when compensation cannot be completed, the claim is left recoverable rather than falsely marked complete.
- Quarantine expiration now commits a `PURGE_CLAIMED` intent before unlinking. Interrupted purges are reconciled on the next maintenance run.
- Quarantine restore validates basenames and compensates a filesystem move if the database commit fails.
- Evidence deletion commits the DB/audit change before unlinking the file; failed physical deletion remains an orphan for controlled reconciliation instead of leaving a DB row that points to a missing file.
- Added an additive Alembic migration for `image_reviews` and `evidence_cleanup_leases`.

### AI image detector integrity and privacy

- Enabled the configured ELA + FFT multi-detector path by default, matching the documented Milestone 3 architecture.
- Updated the disagreement test to assert that conflicting ELA/FFT signals remain `INCONCLUSIVE` rather than forcing a positive or negative authenticity decision.
- Removed the hard-coded neural score behavior. The current neural runner may verify an artifact and its checksum, but it now reports inference unavailable until a real architecture/forward adapter is registered. It no longer emits a fabricated constant `0.50` score.
- Separated preprocessing warnings from quality/disagreement warnings in the API response to avoid duplicated warning arrays.
- Suppressed Pillow TIFF plugin DEBUG logging because it can emit raw EXIF metadata values such as names, GPS-related fields and device serials.
- Updated the neural timeout test to exercise the actual dedicated executor and verify that the permit remains held after the HTTP 504 until the worker finishes.

## Files included

### Modified

- `ai_core/image_intelligence/config.py`
- `ai_core/image_intelligence/detector.py`
- `ai_core/image_intelligence/model_runner.py`
- `ai_core/image_intelligence/preprocessing.py`
- `backend/app/main.py`
- `backend/app/services/evidence_service.py`
- `backend/app/services/image_service.py`
- `backend/tests/test_concurrency_and_event_loop.py`
- `backend/tests/test_evidence_integrity.py`
- `backend/tests/test_image_api.py`
- `backend/tests/test_image_privacy_and_metadata_leakage.py`
- `backend/tests/test_upload_security_and_resource_exhaustion.py`
- `tests/ai/test_image_intelligence.py`
- `tests/ai/test_model_failure_resilience.py`
- `tests/ai/test_model_supply_chain_security.py`

### New

- `backend/app/upload_middleware.py`
- `backend/app/services/file_locking.py`
- `backend/migrations/versions/b61f0a2c9d71_add_image_reviews_and_evidence_cleanup_leases.py`

## Verification performed in this environment

| Check | Result |
|---|---:|
| AI tests: `python -m pytest tests/ai/ -q` | 303 passed |
| Backend tests: `python -m pytest backend/tests/ -q` | 150 passed |
| Python compile check: `python -m compileall -q backend ai_core tests` | Passed |
| Fresh SQLite migration: `python -m alembic upgrade head` | Passed; both migrations applied |
| Alembic schema drift check: `python -m alembic check` | No new upgrade operations detected |
| Focused concurrency/upload/evidence suites | 60 passed before the final added cases; included in the final backend result above |
| Frontend Vitest | Not run: `vitest` was not installed in this execution environment (`npm test -- --run` returned `vitest: not found`) |
| Docker/cgroup and Linux container memory enforcement | Not verified in this environment |

The AI and backend suites were run separately. A combined invocation of both suites in one pytest process timed out; the separate suite runs completed successfully. The full backend suite includes the newly added cleanup and upload middleware tests.

A Pillow `DecompressionBombWarning` is emitted by intentional oversize-image test fixtures; the related tests pass and the input is rejected.

## Important remaining limitations

1. **Neural classification is still not implemented.** The previous constant-score placeholder has been removed. A trained architecture, inference adapter, documented artifact and independent evaluation still belong to Milestone 4.
2. **ELA/FFT remain heuristic triage signals.** This audit does not improve the documented high false-negative limitations on pure AI synthetics and heavily recompressed social-media images.
3. `ImageDetectorConfig.ela_scale` is still present but is not applied in the current ELA score calculation. The historical report's `difference × 15` wording should not be treated as a verified implementation detail; changing that scale requires threshold recalibration and benchmark reruns.
4. Native lock behavior was exercised on the POSIX/Linux test host. The Windows `msvcrt` branch and actual Windows Uvicorn worker deployment were not run here. File-lock coordination assumes all cooperating workers share the same lock directory on a filesystem with correct advisory-lock semantics; independent hosts/replicas require an appropriate shared coordination store.
5. Container cgroup memory enforcement, tmpfs/swap behavior and real deployment load testing remain unverified because Docker was unavailable.
6. The frontend suite remains unverified in this environment. Run it on the Windows development machine after installing the frontend dependencies.
7. The uploaded ZIP included an `.env` file, `.git` data and a private evidence directory. Those were not inspected or included in this correction package. If the uploaded `.env` contained active credentials, rotate them.

## Safe replacement instructions

1. Make a backup copy of your current `SafeHER` folder first.
2. Extract this ZIP directly into the existing repository root, preserving the folder structure and choosing **Replace files** when Windows asks.
3. Do not replace your `.env`, `.git`, private evidence directory, frontend `node_modules`, or model weights; none of them are in this package.
4. From the repository root, run `python -m alembic upgrade head` against your intended database after taking a database backup. This applies the new migration for the two missing tables.
5. Run the AI and backend test commands shown above.
6. From `frontend`, run `npm install` and `npm test` on your Windows development environment.

No Git commit, staging or push was performed.
