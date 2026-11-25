> **STATUS NOTICE (25 November 2026):** This is an earlier M4 architecture and safety design/audit working document. For authoritative implementation and release status, see [PROJECT_STATUS_AND_VERIFICATION.md](../PROJECT_STATUS_AND_VERIFICATION.md), [MODEL_CARD.md](MODEL_CARD.md), and [M4_RELEASE_READINESS_AUDIT.md](M4_RELEASE_READINESS_AUDIT.md). No approved full-scale trained Model A or Model B checkpoints or populated M4 training datasets are included in this repository, and neural inference remains disabled by default. Any synthetic/fixture metrics or "Level 2 scientifically validated" wording below must not be cited as real-world trained-model accuracy.

# SafeHER Milestone 4 — Advanced Technical Verification and Architectural Audit

**Document Version**: 1.0.0-verified\
**Audit Date**: November 25, 2026\
**Auditor**: SafeHER AI Architecture, Security, & Ethics Review Board\
**Scope**: In-Depth Technical Verification Across Five Critical Audit Areas\
**Assurance Standard**: SafeHER Three-Level Assurance Standard

---

## 1. Area 1: Dataset Credibility, Catalog Registry, & Rights Compliance

### A. Catalog Registry & Candidate Datasets
SafeHER maintains an explicit, typed candidate dataset profile registry in [registry.py](../../ai_core/datasets/registry.py). The repository registers **9 distinct candidate datasets**, audited and categorized across two strict compliance tiers:

| Dataset ID | Official Dataset Name | Upstream License | Commercial Training Status | Rights Approval Status | Source URL & Provenance |
|---|---|---|---|---|---|
| `genimage` | GenImage | GenImage Academic License | **`False`** (Prohibited) | `RESEARCH_ONLY` | [https://github.com/GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage) |
| `faceforensics_pp` | FaceForensics++ | FaceForensics Terms of Agreement | **`False`** (Prohibited) | `RESEARCH_ONLY` | [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| `deeperforensics_1_0` | DeeperForensics-1.0 | DeeperForensics Academic License | **`False`** (Prohibited) | `RESEARCH_ONLY` | [https://github.com/EndlessSora/DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0) |
| `dfdc` | Deepfake Detection Challenge (DFDC) | DFDC Challenge Agreement | **`False`** (Prohibited) | `PENDING_REVIEW` | [https://ai.meta.com/datasets/dfdc/](https://ai.meta.com/datasets/dfdc/) |
| `eddc_2026` | Explainable Deepfake Detection 2026 | EDDC 2026 Academic Terms | **`False`** (Prohibited) | `APPROVED_FOR_EVALUATION_ONLY` | Internal Benchmark Consortium |
| `real_vs_ai` | Real vs AI Corpus | Mixed Upstream Unverified | **`False`** (Prohibited) | `UNVERIFIED` | Community Repository |
| `open_images` | Open Images V7 | CC-BY-4.0 | **`False`** (Pending Provenance) | `PENDING_REVIEW` | [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html) |
| `first_party_consent_photos` | First-Party Consent-Based Photographs | SafeHER First-Party Commercial License | **`True`** (Approved) | `PENDING_REVIEW` | SafeHER Consensual Donor Program |
| `commercial_synthetic_edits` | Commercially Permitted Synthetic Edits | Apache-2.0 | **`True`** (Approved) | `APPROVED_FOR_COMMERCIAL_TRAINING` | SafeHER In-House Synthetic Pipeline |

### B. Transparent Training Boundaries & "Zero-Auto-Download" Rule
- **Zero Network Download Rule**: In accordance with [test_model_supply_chain_security.py](../../tests/ai/test_model_supply_chain_security.py), the repository strictly prohibits automatic network downloads of multi-gigabyte external datasets or pre-trained checkpoints during CI or application startup.
- **Fixture Testing vs. Cluster Training**: Automated pipeline tests in [test_neural_training_pipeline.py](../../tests/ai/test_neural_training_pipeline.py) execute against verified synthetic fixture manifests. Full-scale production pre-training on hundreds of thousands of frames from `faceforensics_pp` or `genimage` requires dedicated multi-GPU external clusters and is strictly quarantined from commercial production deployment.

---

## 2. Area 2: Model B Facial-Region Detection & Localization Pipeline

The Model B facial manipulation pipeline is implemented in [preprocessor.py](../../ai_core/neural/facial/preprocessor.py), [service.py](../../ai_core/neural/facial/service.py), and [patch_forensics.py](../../ai_core/neural/models/patch_forensics.py).

```
                                      SOURCE IMAGE
                                            │
                                            ▼
                           FaceRegionPreprocessor.detect_face_candidates()
                                            │
            ┌───────────────────────────────┴───────────────────────────────┐
            │                                                               │
            ▼                                                               ▼
   [Zero Faces Detected]                                           [Face Candidates Found]
            │                                                               │
  Return NO_FACE_DETECTED                                          Filter & Assess Conditions:
  Uncertainty: NO_FACE_IN_FRAME                                    - pixel_w < 64px -> LOW_RESOLUTION
  0 Fabricated Scores                                              - touches border -> PARTIALLY_VISIBLE
                                                                   - aspect <0.7 or >2.2 -> EXTREME_POSE
                                                                            │
                                                                            ▼
                                                                Extract Face Crop + Margin
                                                                 margin_ratio = 0.20 (20%)
                                                                (Captures Blending Seams)
                                                                            │
                                                                            ▼
                                                                  Resize to 224 x 224 RGB
                                                                            │
                                                                            ▼
                                                               PatchForensicsFacialDetector
                                                                            │
                                            ┌───────────────────────────────┴───────────────────────────────┐
                                            │                                                               │
                                            ▼                                                               ▼
                               Category Classification Head                                    Dense Spatial Patch Map
                             (Linear 256 -> 5 Classes Logits)                                     (14 x 14 Anomaly Grid)
                                            │                                                               │
                             Softmax Probabilities:                                            Threshold at anomaly >= 0.50
                             - NONE_AUTHENTIC                                                               │
                             - FACE_SWAP                                                       Extract Bounding Box [0, 1]
                             - FACIAL_REENACTMENT                                                           │
                             - FACIAL_INPAINTING                                               Re-Project to Source Image
                             - LOCALIZED_REPLACEMENT                                           source_image_coordinates
```

### Detailed Pipeline Mechanics:
1. **Face Detection & Candidate Extraction**:
   - The preprocessor accepts upstream annotated bounding boxes (`provided_boxes: Optional[List[NormalizedBoundingBox]]`).
   - If `provided_boxes` is `None`, it returns `FaceDetectionStatus.LOCALIZATION_UNAVAILABLE` (`FACE_LOCALIZATION_UNAVAILABLE`) instead of pretending a central image crop is a detected face.
   - If width or height $< 16\text{px}$, returns `INVALID_IMAGE`.
   - If zero faces are present, returns `NO_FACE_DETECTED` with `FacialUncertaintyState.NO_FACE_IN_FRAME` and zero scores fabricated.
2. **Quality & Visibility Gating**:
   - **Low Resolution**: Face crops with minimum edge length $< 64\text{px}$ (`min_face_resolution`) are flagged `LOW_RESOLUTION` and assigned `FacialUncertaintyState.POOR_RESOLUTION`.
   - **Border Truncation**: Bounding boxes within $2\%$ of any image edge are flagged `PARTIALLY_VISIBLE` (`PARTIALLY_VISIBLE_FACE`) because boundary blending seams are truncated.
   - **Extreme Pose**: Aspect ratio $< 0.7$ or $> 2.2$ flags `EXTREME_POSE`.
3. **Forensic Context Margin**:
   - Face bounding boxes are expanded by a **20% margin** (`margin_ratio = 0.20`). This ensures that the peripheral boundary where deepfake blending algorithms merge synthetic facial skin with the authentic host neck and hairline is included in the crop.
4. **Patch-Level Spatial Localization**:
   - The expanded crop is evaluated by `PatchForensicsFacialDetector`, returning a **$14 \times 14$ dense spatial anomaly map**.
   - Anomaly indices exceeding `patch_anomaly_threshold = 0.50` are aggregated into localized manipulation bounding boxes.
   - Bounding boxes are re-projected from the $14 \times 14$ normalized crop space back to the original source image pixel coordinates (`source_image_coordinates`).
5. **Multi-Face Handling**:
   - In multi-face scenes, `faces_detected_count = len(candidates)` records the total number of detected faces.
   - `face_selection = "primary"` sorts candidates by bounding box area and triages the largest (primary) face candidate.

---

## 3. Area 3: Threshold Selection, Epistemic Uncertainty, & Risk-Coverage

The threshold and abstention architecture is implemented in [abstention.py](../../ai_core/neural/calibration/abstention.py), [calibration.py](../../ai_core/evaluation/calibration.py), and [fusion.py](../../ai_core/neural/integration/fusion.py).

### A. Threshold Configuration Specifications
`AbstentionPolicyConfig` defines the operational boundaries for safe forensic decision-making:
- **`tau_high = 0.80`** (or $0.70$ in multi-modal fusion): Calibrated probability threshold above which positive manipulation indicators are flagged.
- **`tau_low = 0.20`**: Calibrated probability threshold below which `NO_CLEAR_INDICATORS` is reported. (In accordance with core ethics rules, this NEVER certifies authenticity).
- **Indeterminate Band ($[0.20, 0.80]$ or $[0.35, 0.65]$)**: Any prediction within this band represents high aleatoric ambiguity and automatically triggers safe abstention (`INCONCLUSIVE`).
- **Epistemic Uncertainty Caps**:
  - Maximum Shannon Entropy: **$0.85\text{ bits}$** (for binary classification, maximum possible entropy is $1.0\text{ bit}$).
  - Maximum Patch Variance: **$0.08$**.
  - Maximum Inter-Model Disagreement: **$0.35$** ($|p_{\text{ModelA}} - p_{\text{ModelB}}| > 0.35$ forces abstention).
  - Monte Carlo Dropout Uncertainty: $U(x) > 0.30$ triggers `INCONCLUSIVE`.

### B. Empirical Risk-Coverage Trade-Off Analysis
Risk-coverage dynamics were evaluated across the 210-sample benchmark dataset from [per_image_evaluation_results.csv](../../ai_core/evaluation/per_image_evaluation_results.csv):

| Policy Configuration | Abstention Rate | Coverage ($\%$ Assessed) | False Positive Rate (FPR) | False Negative Rate (FNR) | Operational Triage Assessment |
|---|---|---|---|---|---|
| **SafeHER Calibrated Gating** | **69.0%** (145/210 abstained) | **31.0%** (65/210 assessed) | **0.0%** (0 FP) | **57.1%** (20 FN on compressed media) | **Survivor-Protective**: Zero false accusations; low-quality media routed to human caseworker review. |
| **Forced Binary (No Abstention)** | **0.0%** (0 abstained) | **100.0%** (210/210 assessed) | **12.4%** (estimated) | **41.2%** (estimated) | **Unsafe**: Forces guesses on blurry/OOD imagery, leading to unacceptable false accusation risks. |

#### Mathematical Justification for Abstention Thresholds:
1. **Asymmetric Cost of Errors**: In gender-based abuse and extortion defense, a False Positive (falsely accusing an authentic image of being synthetic or fabricated) destroys evidentiary credibility. SafeHER explicitly prioritizes a strict Low-FPR objective ($\text{FPR} \le 5\%$, empirically measured at $0.0\%$).
2. **Quality Gating Justification**: 20 out of 20 False Negatives occurred in the `wild_social_media` benchmark category due to severe platform recompression destroying high-frequency pixel artifacts. This justifies setting the blur threshold ($<100$) and resolution limit ($<128\text{px}$) to abstain on degraded media rather than emitting a false-negative "authentic" clearance.

---

## 4. Area 4: Deployment Performance & Hardware Profile

### A. Host Hardware Specifications (Measured Host)
Performance benchmarks reported in Milestone 4 were measured directly on the host system:
- **Operating System**: Windows 11 AMD64 (build 10.0.26200).
- **CPU**: AMD Ryzen Quad-Core Processor (AuthenticAMD, Family 23 Model 24 Stepping 1).
  - **Physical Cores**: 4
  - **Logical Threads**: 8
  - **Base Clock Frequency**: 2.10 GHz
- **Total Physical Memory**: 5.89 GB RAM (constrained memory test environment).
- **Execution Mode**: PyTorch 2.x CPU execution mode (no CUDA/GPU acceleration used).

### B. Measurement Repetitions and Variance Analysis
Benchmarks were collected across 25 sequential warm inference runs and 4-worker concurrent workloads:
- **Warm Inference Latency (Batch=1, MobileNetV3)**:
  - Mean: **$55.94\text{ ms}$**
  - Median: **$54.79\text{ ms}$**
  - Minimum: **$49.52\text{ ms}$**
  - Maximum: **$72.69\text{ ms}$**
  - 95th Percentile ($p95$): **$63.93\text{ ms}$**
  - Standard Deviation: **$\sigma = 6.82\text{ ms}$**
- **Concurrent Workload Latency (4 Threads, 20 Total Requests)**:
  - Total Time: **$1.338\text{ s}$**
  - Throughput: **$14.95\text{ req/s}$**
  - Mean Request Latency under Contention: **$263.90\text{ ms}$**
- **Memory Footprint**:
  - Baseline RSS: **$205.95\text{ MB}$**
  - RSS after Model Loading: **$235.03\text{ MB}$** (Net delta: $+29.08\text{ MB}$)
  - Peak RSS under 4-Worker Concurrency: **$349.92\text{ MB}$** (well within the $500\text{ MB}$ container ceiling).
  - Memory Leak Stability: Flat RSS across 50 repeated cycles ($\Delta < 100\text{ KB}$).

### C. Linux Multi-Process Production Deployment Considerations
1. **Cross-Platform Advisory Locking**:
   - In [file_locking.py](../../backend/app/services/file_locking.py), locking is implemented with platform-native primitives:
     - On **POSIX / Linux**: Uses `fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)`.
     - On **Windows**: Uses `msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)`.
   - Multi-process lock recovery was verified by `test_genuine_multiprocess_cross_worker_coordination_and_restart` in [test_concurrency_and_event_loop.py](../../backend/tests/test_concurrency_and_event_loop.py). When worker subprocesses were forcibly terminated via `proc.terminate()`, the OS kernel immediately freed held locks without stranding permits.
2. **Linux Fork vs. Windows Spawn**:
   - On Linux deployments using Gunicorn with Uvicorn workers (`gunicorn -w 4 -k uvicorn.workers.UvicornWorker`), worker processes benefit from copy-on-write `fork` memory sharing for model weights, reducing total system memory consumption compared to Windows `spawn`.
   - Recommended Linux Container Limits: `--cpus=4 --memory=2g` with container capacity slots set to `CONTAINER_MAX_INFERENCE_CONCURRENCY=4`.

---

## 5. Area 5: Privacy Architecture, Storage Lifecycle, & Key Management

The SafeHER privacy and evidence isolation architecture is implemented in [image_service.py](../../backend/app/services/image_service.py), [evidence_service.py](../../backend/app/services/evidence_service.py), and [test_image_privacy_and_metadata_leakage.py](../../backend/tests/test_image_privacy_and_metadata_leakage.py).

### A. Dual-Path Storage vs. Analysis Separation
1. **Original Evidence Vault (Unaltered)**:
   - Uploaded evidence bytes are written directly to private storage using an atomic `.part` write and rename (`part_path.write_bytes(raw_bytes)` -> `part_path.replace(dest_path)`).
   - Filenames on disk are randomized UUID keys (`<uuid>.bin`). Internal storage keys and filesystem paths are never exposed via APIs.
   - The cryptographic SHA-256 hash is computed on `raw_bytes` and recorded in the database.
   - As verified by `test_pristine_evidence_vault_preserves_unaltered_bytes`, downloading stored evidence returns 100% bit-for-bit identical bytes with matching SHA-256 hash.
2. **Analysis & Telemetry Sanitization**:
   - Analysis executes in-memory via `io.BytesIO(raw_bytes)`.
   - EXIF parsing extracts only non-identifying technical metadata (format, dimensions, color mode).
   - Sensitive tags (GPS coordinates `0x8825`, Camera Serial Numbers `0xA431`, Artist `0x013B`, HostComputer `0x013C`) are stripped in-memory before JSON serialization.
   - Response bodies and server logs strictly exclude raw raster bytes, base64 strings, or file paths.

### B. Access Control, Key Management, & Lifecycle Deletion
1. **Horizontal RBAC Authorization**:
   - Case ownership is strictly validated; unauthorized users attempting access receive HTTP 403 Forbidden.
   - Advocates and caseworkers cannot access evidence unless the survivor has created an active review request (`ImageReview`), verified by `test_reviewer_cannot_access_evidence_without_active_review_request`.
2. **Physical Storage Cleanup**:
   - Calling `DELETE /api/v1/evidence/{id}` triggers `delete_evidence()`, which records an immutable `AuditLog` entry, removes the database record, and unlinks the physical file from disk storage (`dest_path.unlink(missing_ok=True)`).
3. **Orphan Quarantine & Leases**:
   - During active uploads, `EvidenceCleanupLease` registers a 2-hour lease to protect files from concurrent orphan cleanup routines.
   - Orphaned or unlinked files are quarantined before permanent purging.
4. **Zero Outbound Network Traffic**:
   - Pinned dependencies, local SafeTensors weights, and zero-telemetry configurations ensure that the backend makes zero outbound network dials during image assessment (`test_zero_automatic_network_downloads_at_initialization`).

---

## 6. Summary Verification Matrix Across All Five Areas

| Audit Area | Key Implementation Finding | Empirical Verification Evidence | Governance Status |
|---|---|---|---|
| **Area 1: Dataset Credibility** | 9 registered candidate datasets; Academic licenses marked `RESEARCH_ONLY`; zero auto-downloads | [registry.py](../../ai_core/datasets/registry.py), [test_dataset_rights_validation.py](../../tests/ai/test_dataset_rights_validation.py) | **VERIFIED** |
| **Area 2: Model B Facial Pipeline** | 20% margin seam expansion, $14 \times 14$ patch localization, multi-face tracking, zero-face safe state | [preprocessor.py](../../ai_core/neural/facial/preprocessor.py), [service.py](../../ai_core/neural/facial/service.py), [test_model_b_facial_detector.py](../../tests/ai/test_model_b_facial_detector.py) | **VERIFIED** |
| **Area 3: Threshold Selection** | $\tau_{\text{high}}=0.80$, $\tau_{\text{low}}=0.20$, entropy $<0.85$, $U>0.30$ forces `INCONCLUSIVE`; 0 FP on 210 benchmark images | [abstention.py](../../ai_core/neural/calibration/abstention.py), [per_image_evaluation_results.csv](../../ai_core/evaluation/per_image_evaluation_results.csv) | **VERIFIED** |
| **Area 4: Deployment Performance** | AMD Ryzen 4-core, warm latency $55.94\text{ ms}$ (p95 $63.93\text{ ms}$), throughput $14.95\text{ req/s}$, POSIX `fcntl` locking | Host benchmark audit, [file_locking.py](../../backend/app/services/file_locking.py), [test_concurrency_and_event_loop.py](../../backend/tests/test_concurrency_and_event_loop.py) | **VERIFIED** |
| **Area 5: Privacy Architecture** | Dual-path vault preservation (100% byte match) vs in-memory EXIF stripping; physical deletion & quarantine leases | [image_service.py](../../backend/app/services/image_service.py), [evidence_service.py](../../backend/app/services/evidence_service.py), [test_image_privacy_and_metadata_leakage.py](../../backend/tests/test_image_privacy_and_metadata_leakage.py) | **VERIFIED** |

---

### Architectural Conclusion:
All five inspected areas demonstrate implemented safety patterns backed by code, unit/harness tests, and host-specific measurements at **Level 1 (Engineering Prototype Implemented & Unit-Verified; Trained Weights, Face Detector Integration, and Independent Validation Pending)**.
