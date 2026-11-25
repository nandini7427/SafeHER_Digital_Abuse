# SafeHER M4 — Release Readiness Status (Supersedes Earlier Draft Verdicts)

**Review date:** 25 November 2026

## Verdict

**M4 is an engineering prototype, not a scientifically validated or production-ready neural detector.** The supplied archive contains model/training/inference code but no approved trained Model A/Model B checkpoints or actual populated M4 image datasets. Neural inference remains disabled by default.

Earlier versions of this report that described M4 as “Level 2 scientifically validated” or “production-ready” are superseded. Test harnesses, synthetic fixtures and an implemented training pipeline do not demonstrate real-world model performance.

## Release gate matrix

| Gate | Current status | Evidence / blocker |
|---|---|---|
| Model architectures | Implemented | Source code and architecture unit tests |
| Training pipeline | Implemented, not run at full dataset scale | Synthetic fixtures only; no approved dataset included |
| Model A trained checkpoint | NOT VERIFIED / NOT AVAILABLE | No deployable checkpoint found in supplied archive |
| Model B trained checkpoint | NOT VERIFIED / NOT AVAILABLE | No deployable checkpoint found in supplied archive |
| Dataset licensing and provenance | NOT VERIFIED for any training corpus | No populated dataset register or image corpus included |
| Leakage-resistant split | Utilities implemented | Must be run against real dataset lineage |
| Neural calibration | NOT COMPLETE | No trained-model calibration experiment |
| Neural held-out evaluation | NOT COMPLETE | No trained weights / locked real evaluation set |
| Independent blind evaluation | NOT COMPLETE | No independent evaluator evidenced |
| Secure artifact loading | Implemented and strengthened | Requires verified metadata and approved usage state |
| Backend integration | Implemented | 218 backend tests passed in final Windows release check (including 52 config security tests) |
| Frontend TypeScript & Vite | PASS on Windows | 23 Vitest tests and `tsc && vite build` passed locally; Linux CI verification pending |
| Evidence integrity | Implemented | SHA-256 integrity and access-control tests; not application-level encryption |
| Production deployment | NOT APPROVED | Encryption at rest, deployment configuration, operations and independent review remain |

## Mandatory conditions before a controlled M4 pilot

1. Acquire and legally review actual training and test datasets.
2. Train Model A and Model B on documented datasets; retain complete run logs and approved checkpoints.
3. Freeze model artifacts before final testing.
4. Use identity/source/generator-group isolation and a locked held-out set.
5. Report confusion matrices, per-class precision/recall, false-positive/false-negative rates, abstention coverage and uncertainty.
6. Fit calibration on a separate split; never calibrate on final test data.
7. Evaluate social-media transformations, unseen generators, low-quality inputs and distribution shifts.
8. Conduct independent blind evaluation before making strong performance claims.
9. Complete privacy, encrypted-storage, deployment, rollback and incident-response review.

## Hackathon demo guidance

Demonstrate the existing M1/M2/M3 workflows using synthetic or consent-cleared images and messages. In the M4 panel, clearly show that neural detectors are not enabled until approved trained weights are supplied. It is more credible to demonstrate an honest unavailable state than to show a random or untrained model output.

## Additional final-pass finding: Model B face localization

The supplied Model B preprocessing code does not include a genuine face detector. Earlier code used a central-image crop as a heuristic fallback when no face boxes were provided. A central crop is not face localization and could cause arbitrary image regions to be presented as facial analysis. The final review removes this fallback: Model B now requires verified upstream face boxes or explicit annotations and returns `FACE_LOCALIZATION_UNAVAILABLE` otherwise. Model B evaluation requires annotated `face_bounding_boxes` in each dataset record and rejects unassessed outputs instead of counting them as authentic or manipulated. A production face detector and its independent evaluation remain future requirements.
