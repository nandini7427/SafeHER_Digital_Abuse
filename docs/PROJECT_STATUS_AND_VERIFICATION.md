# SafeHER — Project Status, Audit Findings and Release Readiness

**Audit snapshot:** 25 November 2026\
**Scope:** Supplied SafeHER ZIP and changes made during this review\
**Purpose:** Authoritative status for the replacement project package. Earlier milestone reports containing stronger claims are historical and must not override this document.

## Executive conclusion

SafeHER has a substantial prototype codebase across M1–M4, but the project is **not production-ready and M4 is not scientifically validated as an operational neural detector**. This review strengthened security and corrected integration gaps, but it did not create or invent trained model weights, datasets, calibration evidence, or independent validation.

The current safe operating position is:

- M1 rule/pattern-based threat assessment is available; the optional Toxic-BERT model remains disabled unless a real approved local model is provided.
- M2 authentication and owner-scoped incident case management are available.
- M3 image intake, SHA-256 integrity checks, ELA/FFT and human-review paths are available as prototype functionality. Storage is private by application path and access controls; there is no application-layer encryption-at-rest implementation in the supplied code.
- M4 neural architectures and engineering infrastructure are present, but no genuine trained Model A or Model B checkpoint or actual training dataset was present in the supplied repository. Neural inference remains disabled by default.

## What was verified in this review

### Repository and model artifacts

- No Model A/Model B deployable weights were found in the scanned repository paths (`.pt`, `.pth`, `.safetensors`, `.onnx`, `.ckpt`, `.bin`).
- No actual dataset image corpus or populated dataset manifest was found in the dataset module directory. The dataset schemas, registries, validators and loaders are code, not proof that datasets have been acquired.
- `AI_ENABLE_NEURAL_DETECTOR` defaults to `false`; the example configuration leaves model paths and hashes empty.
- The M4 report itself states that full-scale training has not been run and that local checkpoints were architecture/unit-test fixtures.
- Internal metric harnesses and generated CSVs do not constitute real-world evaluation of trained neural detectors. Do not quote their metrics as Model A or Model B performance.

### Security and correctness changes in this package

- Model A inference no longer permits unweighted/random model initialization to appear as a usable detector through a legacy flag.
- Model A checkpoint loading requires explicit completed-training provenance, architecture match, positive completed epoch, valid training and split manifest hashes, and dataset-version metadata.
- Model B now uses the shared safe checkpoint loader instead of unrestricted `torch.load(..., weights_only=False)`. Missing training metadata cannot silently default to trained, and research-only, pending-review or missing usage-approval states are rejected.
- Secure artifact management now blocks `RESEARCH_RESTRICTED` and `PENDING_REVIEW` artifacts from operational loading. Only `COMMERCIAL_TRAINED` is accepted; `REJECTED` retains its explicit rejection message.
- Secure artifact loading validates that the declared format matches the artifact extension.
- Model B distinguishes `NOT_ASSESSED` from its legacy non-manipulation class and does not return a reassuring classification when the model is unavailable.
- PyTorch checkpoint metadata stores the version as a plain string, supporting safe `weights_only=True` resume loading.
- Training resume recovers the prior best checkpoint and training only finalizes a deployable checkpoint after a completed run with valid provenance.
- Backend case list/create/detail endpoints were added with authentication and owner isolation because the frontend already depends on `/cases`.
- Frontend now has sign-in/register, case management and message-assessment screens; bearer tokens are stored in session storage rather than local storage.
- Frontend copy no longer claims application-level evidence encryption. It describes private storage and integrity verification accurately.
- `.gitignore` now excludes environment secrets, local databases, private evidence, logs, frontend dependencies/build output and local model artifacts.

## Test and build evidence from final release verification

Full local verification was executed on the Windows development host prior to commit:

| Command / suite | Result | Notes |
|---|---:|---|
| `python -m pytest tests/ai/ -q` | **585 passed, 2 skipped** | 587 AI tests collected; 2 Windows unprivileged symlink tests skipped |
| `python -m pytest backend/tests/ -q` | **218 passed** | Includes 52 JWT environment-security tests in `backend/tests/test_config.py`; includes intentional Pillow `DecompressionBombWarning` guard test |
| `npm test -- --run` (in `frontend/`) | **23 passed** | 2 Vitest test files (`ForensicModalityBreakdown.test.tsx`, `SafeImageAnalysisWorkflow.test.tsx`) |
| `npm run build` (in `frontend/`) | **PASS** | `tsc && vite build` compiled 44 modules into `dist/` cleanly |
| `python -m compileall -q ai_core backend tests` | **PASS** | Zero syntax or compilation errors |
| FastAPI OpenAPI & `/health` check | **PASS** | 21 OpenAPI paths registered; `GET /health` returned `200` (`{"status": "healthy"}`) |

These passing test suites verify software engineering, API security, privacy controls, and deterministic scaffold behavior. **They do not establish real-world neural deepfake detection accuracy**, because no trained M4 model weights or external datasets are supplied.

## Known limitations and release blockers

| Area | Current finding | Required before pilot/production |
|---|---|---|
| Model A | Architecture and training framework only; no approved trained artifact | Train on rights-cleared, documented datasets; freeze artifact; independently evaluate |
| Model B | Architecture and inference scaffolding only; no approved trained artifact; no face detector is included or wired into the live pipeline | Add a verified face-localization component or supply reviewed annotations; then train/evaluate on rights-cleared facial-manipulation data |
| Dataset governance | Validators and manifest schemas exist; no actual registered image corpus was included | Verify actual dataset versions, upstream licenses, terms, consent and provenance sample-by-sample |
| Neural calibration | No genuine trained model calibration set/results available | Fit calibration using a separate calibration split from the trained model; preserve final test set |
| Benchmarking | Existing CSV/audit scripts are controlled fixtures, not field evaluation | Evaluate trained checkpoints on locked, representative, held-out data and report per-class errors/coverage |
| Evidence at rest | Private directory, ownership checks and hashes; no app-level encryption at rest | Encrypted volume/storage service, managed keys, access controls and recovery/deletion review |
| Independent evaluation | No third-party blind evaluation | External evaluation with frozen model and unseen data |
| Deployment | Local prototype; hosted production conditions not verified | HTTPS, production DB/secrets, Linux/container tests, backups, monitoring, capacity and incident response |
| Frontend build | Verified on Windows (`npm test -- --run`, `npm run build` passed) | Re-verify `npm ci`, `npm test -- --run`, `npm run build` on target Linux/container CI |

## Important benchmark interpretation

The previously reported 210-row CSV is generated by a controlled evaluation harness and must not be presented as live survivor-case performance or as measured Model A/Model B performance. If its stated counts are used purely as a diagnostic of that harness, 65 of 210 samples received an assessed binary outcome (30.95% coverage); among the 35 assessed positive examples, 15 were true positives and 20 false negatives (42.86% recall on assessed positives). This is not a neural-model performance estimate and should not be used in a pitch as evidence of detection accuracy.

## GitHub safety checklist

Before pushing the replacement repository:

1. Rotate any database, JWT or API credentials that appeared in the original ZIP's `.env` file. The values are intentionally not reproduced in this report.
2. Create a new `.env` locally from `.env.example`; never commit it.
3. Inspect `git status --short`, `git diff --check`, `git check-ignore -v .env frontend/node_modules frontend/dist private_evidence`.
4. Confirm no private evidence, actual datasets, model weights, logs, local databases, build output or credentials are staged.
5. Run the complete test suites and frontend build on the development machine after clean dependency installation.
6. Commit only after manually reviewing the staged file list.

## Release verdict

**Prototype / hackathon demo readiness: CONDITIONAL.** Use synthetic or consent-cleared examples, clearly label neural inference as unavailable until approved trained weights exist, and avoid real confidential survivor evidence.

**M4 scientific validation: NOT COMPLETE.** Architecture and code tests do not establish detection performance.

**Production readiness: NOT APPROVED.** Do not deploy for real survivor evidence until the blockers above are addressed and independently reviewed.

## Additional final-pass correction: Model B face-localization honesty

A final review found that Model B's preprocessor previously used a central image crop when no face boxes were supplied. This was not genuine face detection. The replacement removes that fallback and returns `FACE_LOCALIZATION_UNAVAILABLE` unless verified face boxes or annotations are supplied. Model B evaluation now requires explicit annotated face boxes and rejects unavailable/`NOT_ASSESSED` predictions rather than silently counting them as negative. A genuine, independently validated face-localization model is still not included or wired into the live pipeline. The frontend now distinguishes `FACE_LOCALIZATION_UNAVAILABLE` from a verified `NO_FACE_DETECTED` outcome and displays the explicit limitations rather than claiming there are zero faces.
