# SafeHER final re-audit — 25 November 2026

## Scope

A final focused re-audit was performed on the corrected replacement source tree after the first packaging pass. This was an additional review of the M4 face-localization path, artifact-approval enforcement, UI status semantics, test accounting and GitHub packaging exclusions.

## Additional defects found and corrected

1. **Model B did not have a real face-localization component.** Its preprocessor previously substituted a central crop when no face boxes were supplied. That is not face detection. The fallback was removed. Missing verified face boxes now produce `FACE_LOCALIZATION_UNAVAILABLE`; a verified empty face result remains `NO_FACE_DETECTED`.
2. **Model B evaluation could treat unassessed samples as negative.** Evaluation now counts unavailable/`NOT_ASSESSED` samples as abstentions, not as authentic or manipulated. Metrics use assessed samples and report coverage; metrics are `None` where no valid assessed samples exist.
3. **Independent Model B evaluation now requires explicit face-box annotations** in the dataset record metadata. Missing face annotations fail the evaluation sample instead of silently analyzing the image center.
4. **The frontend previously displayed a missing face-localization result as “No Face in Frame” / “Zero detected.”** The API now carries face-localization status and limitations, and the frontend distinguishes unavailable localization from a verified no-face result. It also displays “Not assessed” instead of a reassuring authenticity-like fallback.
5. **The shared checkpoint loader did not enforce an explicit permitted-usage state.** The loader now requires `approved_usage_state=COMMERCIAL_TRAINED`; missing, research-restricted, pending-review and rejected states are refused. The training pipeline records the approval state from its manifest safety audit, while the checkpoint manager defaults to `PENDING_REVIEW` when no approval is supplied.
6. **Model B now classifies an artifact-approval failure as an untrained/rejected model state** and does not leave a partially initialized service operational.
7. **The test collection count was updated** after adding final regression coverage.

## Final-retry verification evidence

| Verification | Result |
|---|---:|
| Model A + Model B + independent evaluation + supply-chain tests | 73 passed |
| Neural training + secure artifact + rollback tests | 39 passed |
| Neural/forensic integration tests | 13 passed |
| Backend neural inference + image API + case API tests | 30 passed |
| Python bytecode compilation (`compileall`) | PASS |
| Frontend TypeScript check (`tsc --noEmit`) | PASS |
| Latest Python collection | 753 collected (587 AI + 166 backend) |

These are targeted test runs, not a claim that all 753 collected Python tests passed. Full AI and backend suite attempts exceeded the execution limit in this review environment. Frontend Vitest and the Vite production build remain unverified here because the uploaded `node_modules` tree was Windows-specific; run `npm ci`, `npm test -- --run` and `npm run build` on the target Windows development machine.

## Remaining blockers — do not overclaim

- No approved full-scale trained Model A or Model B weights were included in the source archive.
- No populated, rights-cleared M4 image dataset or item-level license/provenance inventory was included.
- No genuine trained-model performance results or calibration results are established by this package.
- No face detector/localizer is included or wired into the live Model B pipeline. Reviewed face annotations or a verified upstream face-localization model are required before facial manipulation analysis can run.
- No independent blind evaluation has been performed.
- Evidence storage uses private filesystem controls and SHA-256 integrity checks; application-level encryption at rest and managed key lifecycle are not implemented.
- Production deployment, Linux/container behavior, secrets management, backups, incident response and independent privacy/security review remain outstanding.

Keep `AI_ENABLE_NEURAL_DETECTOR=false`. Never use survivor evidence for model training or evaluation. Do not use real intimate imagery for a hackathon demo; use synthetic, public-domain or consent-cleared non-sensitive examples.

## Package hygiene

The replacement ZIP excludes `.env`, `.git`, private evidence, local databases, logs, Python caches, `node_modules`, frontend build output and local model binaries. The original archive contained an `.env`; rotate any credentials that were present in it before publishing the project.

## Release verdict

**Hackathon prototype: conditional.** The corrected project has a clearer and safer end-to-end prototype, but the demo must be transparent that M4 neural models are not trained/approved and that Model B currently has no wired face-localization detector.

**Scientific validation: not complete. Production readiness: not approved.**
