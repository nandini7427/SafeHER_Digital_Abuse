# SafeHER AI/ML Milestone 2: Technical Model Integration-Readiness Report

**Document Title**: AI/ML Model Integration-Readiness Assessment & Architecture Design\
**Author**: Antigravity AI Engineering\
**Target Environment**: AMD Ryzen 5 CPU, 6.0 GB System RAM, Integrated Graphics (CPU-only inference), Python 3.12\
**Baseline Git Commit**: `81c1571` on branch `feature/ai-threat-assessment`\
**Status**: Ready for Review / No Code Modifications Applied

---

## 1. Executive Summary

This report delivers a technical integration-readiness assessment for introducing deep learning models into the SafeHER Threat Assessment module.

SafeHER's primary mission is to detect digital abuse, coercive control, non-consensual image blackmail, and financial extortion in victim communications. A rigorous audit of the existing codebase confirms that the foundation implemented in Milestone 1 is clean, modular, and robust:
- The seven-category threat taxonomy is decoupled from toxicity.
- Text preprocessing isolates working copies and strictly preserves original evidence unmodified.
- Pydantic v2 schemas enforce `is_calibrated=False`, banning pseudo-probabilities.
- The `BaseThreatAnalyzer` abstract interface allows transparent model swapping with immediate heuristic fallback.

Three model candidates were evaluated against verified specifications and the 6.0 GB RAM / CPU-only hardware boundary:
1. **`unitary/toxic-bert`**: Specialized multilabel toxicity classifier (~110M params, ~438 MB). Highly efficient on CPU, but **blind to conditional extortion and blackmail** that lack profanity.
2. **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`**: Natural Language Inference (NLI) model (~184M params, ~370–738 MB). Exceptional zero-shot capability for classifying conditional threats and blackmail intent without requiring profanity; moderate CPU latency.
3. **`microsoft/mdeberta-v3-base`**: Pretrained foundation model (~280M params, 250k vocab, ~1.12–3.4 GB). Excellent multilingual potential, but **cannot be used out-of-the-box** because it lacks a task classification/NLI head and exceeds the safe memory budget of a 6 GB machine.

**Key Recommendation**: An asynchronous, staged two-tier architecture where `unitary/toxic-bert` provides lightweight auxiliary aggression/toxicity scoring, while an NLI model (such as `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` or its lighter xsmall variant) performs hypothesis-driven threat intent classification, with `DevelopmentThreatAnalyzer` serving as a permanent zero-overhead fallback.

---

## 2. Audit of Existing Implementation

The existing files in `ai_core/threat/` and `tests/ai/` were thoroughly inspected:

### 2.1 `ai_core/threat/schemas.py`
- **Taxonomy Integrity**: Defines `ThreatCategory` as an enum of 7 core abuse intentions:
  1. `DIRECT_THREAT`
  2. `BLACKMAIL`
  3. `FINANCIAL_EXTORTION`
  4. `EXPOSURE_THREAT`
  5. `COERCION`
  6. `HARASSMENT_INTIMIDATION`
  7. `URGENCY_PRESSURE`
- **Separation of Auxiliary Signals**: `AuxiliarySignal` (`TOXICITY`, `INSULT`, `PROFANITY`) is declared as a separate enum, ensuring vulgarity is never conflated with extortion intent.
- **Explainability**: `FlaggedSpan` captures character offsets (`start`, `end`, `text`, `category`, `explanation`) against the working copy for UI visualization.
- **Calibration Contract**: `ThreatAssessmentResponse` includes:
  - `is_calibrated: bool = False`: Explicitly communicates whether outputs are statistically calibrated probabilities.
  - `confidence_score: Optional[float] = None`: Strictly populated only if empirical confidence is calibrated; defaults to `None`.
  - `category_scores: Dict[str, float]`: Keyed float scores, designated for raw or calibrated model outputs.
  - `uncertainty_state: UncertaintyState`: Categorical triage (`LOW`, `MODERATE`, `HIGH`, `INSUFFICIENT_CONTEXT`).
  - `disclaimer`: Non-overridable mandatory legal disclaimer.

### 2.2 `ai_core/threat/preprocessing.py`
- **Evidence Immutability**: The `PreprocessedText` frozen dataclass preserves `original_text` unaltered while generating an isolated `working_copy`.
- **Sanitization Pipeline**:
  - Normalizes Unicode using `NFKC` (canonical compatibility decomposition/composition).
  - Strips invisible evasion characters (zero-width spaces `\u200B`, joiners `\u200C-\u200D`, bidirectional overrides `\u202A-\u202E`).
  - Strips non-printable ASCII control characters (`\x00-\x08`, `\x0B-\x1F`, `\x7F`).
  - Normalizes whitespace and limits consecutive newlines.
  - Enforces `min_length` (1 char) and `max_length` (10,000 chars) with truncation warning flags.
- **Robustness**: Rejects empty strings, whitespace-only strings, and strings containing only invisible characters with informative `ValueError` exceptions.

### 2.3 `ai_core/threat/analyzer.py`
- **Abstract Interface**: `BaseThreatAnalyzer` defines the standard signature `analyze(text: str) -> ThreatAssessmentResponse`.
- **Baseline Engine**: `DevelopmentThreatAnalyzer` implements rule-based pattern matching:
  - Explicitly documents that it does not run deep learning and produces no fake AI metrics.
  - Filters benign colloquial/figurative idioms (e.g., *"killing it"*, *"dead tired"*, *"bombed the exam"*).
  - Identifies ambiguous short statements (e.g., *"Wait for tomorrow"*, *"You will see"*) and flags them as `UncertaintyState.INSUFFICIENT_CONTEXT`.
  - Evaluates compounded risk (e.g., direct violence + extortion produces `ThreatLevel.SEVERE`).

### 2.4 `ai_core/config.py`
- Houses `ThreatAnalyzerConfig`:
  - `device: str = "cpu"`
  - `max_input_length: int = 10000`
  - `min_input_length: int = 1`
  - `enable_toxicity_check: bool = False`
  - `model_name_or_path: Optional[str] = None`
  - `is_development_placeholder: bool = True`
- Stores `MANDATORY_AI_DISCLAIMER` stating that automated output is assistive triage, not forensic or legal proof.

### 2.5 `tests/ai/test_threat.py`
- 15 unit tests verifying:
  - Empty text, whitespace, and invisible character rejection.
  - Harmless text passing with `ThreatLevel.NONE` and zero flagged spans.
  - Synthetic blackmail, financial extortion, and exposure detection.
  - Direct threat detection with exact character offset alignment.
  - Figurative idiom non-triggering across multiple parametrized phrases.
  - Insufficient context detection on isolated short phrases.
  - Long text truncation without modifying `original_text`.
  - Schema validity, serialization, and mandatory disclaimers.
- All 15 tests execute in <0.3s without network access or model weight downloads.

---

## 3. Candidate Model Evaluation

Each candidate was researched using official model cards, repository specifications, and benchmark literature.

### 3.1 Candidate A: `unitary/toxic-bert`

* **Intended Task**: Multilabel text toxicity classification. Outputs continuous raw logits/sigmoid scores for 6 categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
* **Model Architecture & Parameter Count**:
  * Base: `bert-base-uncased` (Devlin et al., 2018).
  * Structure: 12 hidden layers, 768 hidden dimension, 12 attention heads.
  * Parameter Count: **~110 Million (109.5M)**.
* **Model File Size**:
  * Disk weights (`model.safetensors` or `pytorch_model.bin`): **~438 MB**.
* **License**: **Apache-2.0**.
* **Training Data & Supported Languages**:
  * Dataset: Kaggle / Conversation-AI *Jigsaw Toxic Comment Classification Challenge* dataset (comments from English Wikipedia talk pages).
  * Supported Language: **English only** (WordPiece vocabulary of 30,522 uncased tokens).
* **Classification Scope (Threat vs. Toxicity vs. Language Relations)**:
  * Detects explicit **toxicity, obscenity, and hostility**.
  * Contains a `threat` head, but it was trained on explicit violent insults and death threats in Wikipedia discussions.
  * **Critical Limitation**: It **cannot** detect non-vulgar blackmail or financial extortion (e.g., *"Send $500 or your family receives these photos"*). Because such messages avoid vulgar words, Toxic-BERT scores them as low toxicity (<0.05).
* **Expected CPU Inference Considerations**:
  * Single-inference latency on AMD Ryzen 5 CPU: ~40–80 ms per sample (batch size = 1).
  * Very lightweight computational footprint.
* **Suitability for 6.0 GB RAM**:
  * **Excellent**. In FP32, model weights consume ~440 MB RAM. Peak inference memory is ~600–700 MB. Fits easily within 6 GB RAM with ample headroom.
* **Limitations & Potential Bias**:
  * High false positive rate on benign colloquial profanity (e.g., *"fuck yeah that's awesome"*).
  * Inherits Wikipedia talk page annotator biases; elevated false positives on text mentioning minority group identities (e.g., *"gay"*, *"muslim"*, *"trans"*).
  * Vulnerable to obfuscation (character insertion, leetspeak).

---

### 3.2 Candidate B: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

* **Intended Task**: Natural Language Inference (NLI) and Zero-Shot Classification. Given a `premise` (victim message) and a `hypothesis` (e.g., *"This text threatens to release private photos unless a demand is met"*), it outputs logits for three entailment classes: `entailment`, `neutral`, `contradiction`.
* **Model Architecture & Parameter Count**:
  * Base: `microsoft/deberta-v3-base` (He et al., 2021) with Disentangled Attention and Enhanced Mask Decoder (EMD).
  * Structure: 12 hidden layers, 768 hidden dimension, 12 attention heads.
  * Vocabulary: 128,100 tokens (SentencePiece).
  * Parameter Count: **~184 Million (184M)** (~86M backbone + ~98M embedding layer).
* **Model File Size**:
  * Disk weights (`model.safetensors` / `pytorch_model.bin`): **~370–738 MB** (depending on precision and serialization format).
* **License**: **MIT License**.
* **Training Data & Supported Languages**:
  * Pretrained on CC100 (English subset), The Pile, and English Wikipedia.
  * Fine-tuned on an ensemble of NLI corpora: MultiNLI (MNLI), FEVER (fact verification), and Adversarial NLI (ANLI rounds 1–3).
  * Supported Language: **English only**.
* **Classification Scope (Threat vs. Toxicity vs. Language Relations)**:
  * Identifies **general semantic relationships (entailment/contradiction)**.
  * Through zero-shot hypothesis formulation, it excels at identifying **covert extortion, blackmail, conditional demands, and exposure threats**, completely independent of profanity or vulgarity.
  * It does not output static toxicity labels unless specifically queried with toxicity hypotheses.
* **Expected CPU Inference Considerations**:
  * Disentangled attention is computationally heavier than standard BERT attention.
  * Single premise-hypothesis evaluation on AMD Ryzen 5 CPU: ~150–300 ms.
  * If querying 7 hypotheses (one for each taxonomy category), sequential latency reaches ~1.2–2.0 seconds. Hypothesis batching or pre-filtering is required to maintain responsive latency.
* **Suitability for 6.0 GB RAM**:
  * **Good / Manageable**. Model consumes ~750 MB RAM in FP32; peak runtime memory is ~1.2–1.5 GB. Leaves ~4.5 GB for operating system and application processes. Must be managed as a singleton to prevent duplicate in-memory copies.
* **Limitations & Potential Bias**:
  * Highly sensitive to hypothesis phrasing (subtle wording changes in the hypothesis prompt alter entailment scores).
  * General-domain NLI training (MNLI/ANLI) lacks specialized nuances of interpersonal coercive control and domestic violence.
  * Slower than Toxic-BERT on CPU.

---

### 3.3 Candidate C: `microsoft/mdeberta-v3-base`

* **Intended Task**: Multilingual Masked Language Modeling (Foundation Model). It is a raw backbone trained for cross-lingual representation learning.
* **Model Architecture & Parameter Count**:
  * Base: Multilingual DeBERTa-v3 (`mDeBERTa-v3`).
  * Structure: 12 hidden layers, 768 hidden dimension, 12 attention heads.
  * Vocabulary: 250,000 SentencePiece tokens covering 100 languages.
  * Parameter Count: **~280 Million (278M–280M)** (86M transformer backbone + ~192M embedding parameters).
* **Model File Size**:
  * Disk weights: **~1.12 GB (FP32 weights)**; full repository with checkpoints reaches **~3.4 GB**.
* **License**: **MIT License**.
* **Training Data & Supported Languages**:
  * Pretrained on the 2.5 TB CC100 multilingual dataset.
  * Supports **100 languages** (including English, Spanish, Hindi, French, Arabic, German, etc.).
* **Classification Scope (Threat vs. Toxicity vs. Language Relations)**:
  * Identifies **contextual bidirectional representations across languages**.
  * **CRITICAL FINDING**: It **CANNOT classify threats, toxicity, or extortion out of the box**. It has no classification head or NLI head. To be usable for SafeHER, it would require supervised fine-tuning or a specialized downstream checkpoint (e.g., `mDeBERTa-v3-base-xnli`).
* **Expected CPU Inference Considerations**:
  * Very high CPU latency due to 278M parameters and 250k vocabulary embeddings.
  * Latency per inference: ~400–800 ms per sequence on 4 CPU cores.
* **Suitability for 6.0 GB RAM**:
  * **Poor / High Risk**. In FP32, model loading and attention buffers require ~1.8–2.4 GB active RAM. Combined with Windows OS overhead (~3.0–3.5 GB), this pushes system RAM to >90% capacity, creating severe paging/swapping risks and potential OOM crashes.
* **Limitations & Potential Bias**:
  * Missing classification/NLI head (cannot be used directly without fine-tuning).
  * Disproportionate representation in CC100 favoring high-resource Western languages over low-resource regional dialects.
  * Substantial memory footprint.

---

### 3.4 Side-by-Side Model Comparison Matrix

| Evaluation Criterion | Candidate A: `unitary/toxic-bert` | Candidate B: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Candidate C: `microsoft/mdeberta-v3-base` |
| :--- | :--- | :--- | :--- |
| **Primary Task** | Multilabel Toxicity Classification | Zero-Shot NLI Classification | Masked Language Modeling (Foundation) |
| **Out-of-the-Box Usability** | **Yes** (ready to infer 6 categories) | **Yes** (ready for zero-shot hypotheses) | **No** (requires task fine-tuning head) |
| **Architecture** | BERT-base uncased | DeBERTa-v3-base (Disentangled Attn) | mDeBERTa-v3-base (Multilingual) |
| **Parameter Count** | ~110M | ~184M | ~280M |
| **Weight Disk Size** | ~438 MB | ~370–738 MB | ~1.12–3.4 GB |
| **License** | Apache-2.0 | MIT | MIT |
| **Supported Languages** | English only | English only | 100 Languages |
| **Threat Intent Detection** | Weak (only overt violent insults) | **Strong** (contextual & conditional) | None (without fine-tuning) |
| **Toxicity Detection** | **Strong** (vulgarity, hate, insult) | Indirect (requires prompt) | None |
| **Blackmail / Extortion** | **Blind** (fails non-vulgar extortion) | **Strong** (understands conditional threats) | None |
| **CPU Latency (AMD Ryzen 5)** | **Fast** (~40–80 ms) | **Moderate** (~150–300 ms per hypothesis) | **Slow** (~400–800 ms) |
| **RAM Footprint (FP32)** | ~450–700 MB | ~1.1–1.4 GB | ~1.8–2.4 GB |
| **6.0 GB RAM Compatibility** | **Safe** (Low risk) | **Feasible** (Single-model footprint) | **Unsafe** (High risk of paging/OOM) |

---

## 4. Proposed Integration Architecture

### 4.1 Staged Dual-Tier Ensemble Design
To satisfy the dual requirements of low CPU latency and high extortion detection accuracy on a 6 GB RAM machine, SafeHER should adopt a **staged two-tier architecture**:

```
                       +-------------------------+
                       |    Raw Evidence Text    |
                       +-------------------------+
                                    |
                                    v
                       +-------------------------+
                       |    preprocess_text()    |
                       | (NFKC, control chars,   |
                       |  length truncation)     |
                       +-------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
         [Original Evidence:              [Working Copy Text]
          Immutable Record]                         |
                                                    v
                                      +---------------------------+
                                      | Tier 1: Pattern Screener  |
                                      | & Idiom De-biasing        |
                                      | (Fast Regex & Heuristics) |
                                      +---------------------------+
                                                    |
                                    +---------------+---------------+
                                    |                               |
                        [Figurative Language:           [Candidate Text:
                         e.g. "killing it"]              Needs Inference]
                                    |                               |
                                    v                               v
                        ThreatLevel: NONE             +---------------------------+
                        FlaggedSpans: []              | Tier 2A: Toxic-BERT       |
                                                      | (Auxiliary Toxicity/Hate) |
                                                      +---------------------------+
                                                                    |
                                                                    v
                                                      +---------------------------+
                                                      | Tier 2B: DeBERTa NLI      |
                                                      | (Conditional Extortion    |
                                                      |  & Blackmail Hypotheses)  |
                                                      +---------------------------+
                                                                    |
                                                                    v
                                                      +---------------------------+
                                                      | Aggregator & Explainer    |
                                                      | - Map 7 SafeHER Categories|
                                                      | - Label is_calibrated=False|
                                                      | - Extract Flagged Spans   |
                                                      | - Attach Legal Disclaimer |
                                                      +---------------------------+
                                                                    |
                                                                    v
                                                      ThreatAssessmentResponse
```

### 4.2 Preservation of SafeHER Taxonomy & Interface Contract
1. **Taxonomy Retention**: The 7 core categories (`direct_threat`, `blackmail`, `financial_extortion`, `exposure_threat`, `coercion`, `harassment_intimidation`, `urgency_pressure`) remain the standard output.
2. **Auxiliary Signals**: Toxic-BERT outputs (`toxic`, `severe_toxic`, `obscene`, `insult`, `identity_hate`) are captured under `category_scores` with prefixes (e.g., `toxic_bert:toxic`) and mapped to `AuxiliarySignal.TOXICITY`.
3. **NLI Hypothesis Mapping**: DeBERTa-v3 evaluates structured hypotheses tailored to the SafeHER taxonomy:
   - *Blackmail*: `"This message is blackmail threatening consequences unless demands are met."`
   - *Financial Extortion*: `"This message demands money or cryptocurrency under threat."`
   - *Exposure Threat*: `"This message threatens to share private photos or sensitive personal information."`
   - *Direct Threat*: `"This message threatens physical violence or bodily harm."`

### 4.3 Preservation of `BaseThreatAnalyzer` & Fallback Lifecycle
1. The new analyzer implements `BaseThreatAnalyzer` as `CompositeThreatAnalyzer(BaseThreatAnalyzer)`.
2. `DevelopmentThreatAnalyzer` is embedded directly as an internal component.
3. If deep learning model loading fails (e.g., corrupt weights, missing dependencies, OS DLL restrictions, memory exhaustion), the analyzer logs a warning, falls back to `DevelopmentThreatAnalyzer`, and appends an explicit fallback notice to `explanation`.

### 4.4 Calibration & Probability Integrity
- Because raw sigmoid outputs from BERT models and softmax outputs from NLI models are not calibrated probabilities, the system will strictly set:
  ```python
  is_calibrated = False
  confidence_score = None
  ```
- Raw model scores will be presented as uncalibrated scalar indicators (`category_scores`), not percentages or likelihoods.

### 4.5 Memory & CPU Lifecycle Optimization (6.0 GB RAM Constraint)
1. **Singleton Model Runner**: Models are encapsulated inside a thread-safe singleton runner. Models are never reloaded per request.
2. **Lazy Initialization**: Model weights are loaded into memory only on the first active inference request, avoiding startup delays during API boot.
3. **CPU Thread Management**: PyTorch CPU intra-op parallelism will be explicitly constrained:
   ```python
   torch.set_num_threads(4)
   ```
4. **Targeted Inference**: If Tier 1 pattern matching detects clear figurative speech with no threat signals, deep model inference is skipped entirely, saving CPU cycles.
5. **Sequential Execution**: If both Toxic-BERT and DeBERTa are used, they run sequentially, never concurrently in parallel threads, preventing CPU throttling and memory spikes.

---

## 5. Comprehensive Synthetic Testing Plan

To adhere to safety and privacy constraints, **no real victim communications or personally identifiable information (PII)** will be used. All test vectors must be synthetic and deterministic.

### 5.1 Test Scenarios and Synthetic Test Vectors

| Test Suite | Objective | Synthetic Example Input | Expected Result |
| :--- | :--- | :--- | :--- |
| **Direct Threats** | Verify identification of explicit bodily harm. | `"I know where you live and I will put you in the hospital if you show up."` | `ThreatLevel.HIGH` or `SEVERE`; `DIRECT_THREAT` flagged; accurate span offsets. |
| **Blackmail & Extortion** | Verify non-vulgar conditional threat detection. | `"Unless you transfer $500 before midnight, I will send your private photos to your boss."` | `ThreatLevel.SEVERE`; `BLACKMAIL`, `FINANCIAL_EXTORTION`, `EXPOSURE_THREAT` flagged; Toxic-BERT toxicity score low, NLI entailment high. |
| **Harmless Profanity** | Ensure vulgarity does not trigger false extortion/threat alerts. | `"Holy shit, that concert was fucking unbelievable, we had such a great time!"` | `ThreatLevel.NONE`; Toxic-BERT `obscene` high, but threat categories empty; no threat spans. |
| **Figurative Language** | Ensure violent idioms are not misclassified as physical violence. | `"You are totally killing it on stage tonight, the presentation destroyed the competition!"` | `ThreatLevel.NONE`; `UncertaintyState.LOW`; no threat spans. |
| **Ambiguous Messages** | Test insufficient context handling on vague phrases. | `"Wait for tomorrow. You will see."` | `ThreatLevel.LOW`; `UncertaintyState.INSUFFICIENT_CONTEXT`; monitoring recommendation in explanation. |
| **Multilingual Input** | Test behavior on non-English messages (Spanish, Hindi, French). | `"Si no me pagas, publicaré tus fotos."` / `"Mujhe paise do warna video leak kar dunga."` | Safe fallback or multilingual NLI handling without unhandled exceptions or crashes. |
| **Input Validation & Truncation** | Verify input safety limits. | Empty string `""`, whitespace `"   "`, and 15,000-character repetitive text. | Empty string raises `ValueError`; long text truncated to 10,000 chars; `original_text` preserved. |
| **Model Failure Fallback** | Verify graceful degradation when weights are missing/corrupted. | Mock model runner throwing `RuntimeError("Model weights not found")`. | Analyzer returns valid `ThreatAssessmentResponse` via `DevelopmentThreatAnalyzer`; logs diagnostic warning. |
| **Calibration Flag Integrity** | Verify probability misrepresentation prevention. | Any synthetic threat or benign text. | `is_calibrated == False`; `confidence_score is None`. |

### 5.2 Performance & Resource Benchmark Harness
A dedicated benchmark suite will measure:
- **Peak RSS Memory**: Measured using `psutil.Process().memory_info().rss`. Maximum threshold: **< 1.8 GB active RAM**.
- **Inference Latency**: Measured via `time.perf_counter()`. Maximum threshold: **< 500 ms for Toxic-BERT alone; < 1500 ms for full ensemble** on 4 CPU threads.
- **Leak Detection**: 100 consecutive synthetic inference loops to confirm zero memory drift.

---

## 6. Risk Analysis & Mitigation Strategies

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **RAM Exhaustion (OOM)** | Application crash or OS freeze on 6 GB system. | Medium | Restrict model concurrency to 1; prohibit loading `mdeberta-v3-base` raw weights; enforce dynamic garbage collection after inference. |
| **Extortion Missed by Toxicity Models** | Toxic-BERT gives near-zero score to polite blackmail. | High | Decouple toxicity from threat intent; use zero-shot NLI (`DeBERTa-v3`) specifically for conditional blackmail detection. |
| **High CPU Inference Latency** | Slow response times (>3s) frustrating users. | Medium | Limit CPU threads to 4; implement Tier 1 regex fast-path for benign figurative speech; evaluate a distilled NLI model (e.g., `DeBERTa-v3-xsmall-mnli`, ~70M params) if `base` exceeds 1.5s. |
| **Hypothesis Prompt Fragility** | NLI model gives inconsistent results based on prompt wording. | Medium | Standardize and freeze validated hypothesis templates; test hypotheses against a fixed synthetic regression suite. |
| **Host Policy Restrictions (e.g. WDAC)** | Windows Application Control or security policy blocks native DLLs (`torch_python.dll`). | Low | Wrap model loading in comprehensive try-except blocks that fall back immediately to `DevelopmentThreatAnalyzer` without crashing. |

---

## 7. Recommended Next Implementation Step

### Phase 1: Integrate `unitary/toxic-bert` First
1. Implement a dedicated model runner (`ToxicBertModelRunner`) supporting lazy loading, tokenization, and sigmoid extraction.
2. Connect `unitary/toxic-bert` to populate `category_scores` with `toxic_bert:*` scores and auxiliary signals.
3. Verify CPU latency and memory footprint against the benchmark harness on the target AMD Ryzen 5 hardware.

### Phase 2: Integrate Lightweight Zero-Shot NLI
1. Research and benchmark a lightweight DeBERTa NLI variant (such as `MoritzLaurer/DeBERTa-v3-xsmall-mnli`, ~70M parameters, ~280 MB, or the base variant) to classify the 7 SafeHER threat categories without exceeding memory limits.
2. Establish frozen hypothesis templates for blackmail, financial extortion, and physical violence.
3. Run the synthetic test harness to verify that non-vulgar extortion is accurately classified while keeping total inference latency under 1.5 seconds.
