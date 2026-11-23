# Pretrained AI Model Candidate Evaluation — SafeHER Threat Assessment

This document provides a rigorous architectural, computational, and hardware-aware evaluation of candidate pretrained models for the SafeHER Threat Assessment engine.

---

## 1. Candidate Comparison Matrix

| Attribute | Candidate 1: `unitary/toxic-bert` | Candidate 2: `microsoft/deberta-v3-small` | Candidate 3: `MoritzLaurer/DeBERTa-v3-xsmall-mnli` |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | `bert-base-uncased` (12 layers, 768 hidden, 12 heads, 110M params) | Disentangled Attention Transformer (6 layers, 768 hidden, 44M params) | Lightweight Cross-Encoder NLI (4 layers, 384 hidden, 22M params) |
| **Model Size (Disk)** | ~438 MB | ~175 MB | ~88 MB |
| **Runtime Memory (RAM)** | ~500–650 MB peak | ~320–420 MB peak | ~180–260 MB peak |
| **Inference Latency (CPU)**| ~35–65 ms / sample | ~40–75 ms / sample | ~20–35 ms / hypothesis |
| **Required Dependencies** | `torch` (CPU), `transformers`, `tokenizers` | `torch` (CPU), `transformers`, `sentencepiece` | `torch` (CPU), `transformers`, `sentencepiece` |
| **License** | Apache 2.0 | MIT License | MIT License |
| **Training Dataset** | Jigsaw Toxic Comment Classification Challenge | CC100 pretraining (fine-tunable on abuse sets) | MultiNLI, SNLI, ANLI, Fever |
| **Classification Task** | Multilabel toxicity detection (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) | Contextual sequence classification across intent categories | Zero-shot premise-hypothesis entailment inference |
| **Multilingual Support** | English only (`uncased` vocabulary) | English only (requires `mDeBERTa-v3` for multilingual, which is ~1.1 GB) | English only (requires `xlm-roberta-base-finetuned-nli` ~1.1 GB for multilingual) |
| **Critical Limitations** | **Not a blackmail or extortion classifier.** Cannot recognize conditional extortion, ransom demands, or non-consensual image distribution threats. | Requires task-specific supervised fine-tuning before deployment; cannot be used zero-shot out-of-the-box. | Sequential multi-hypothesis testing scales linearly with category count ($O(N)$). Sensitive to hypothesis prompt formulation. |

---

## 2. Fundamental Architectural Distinction: Toxicity vs. Threat Intent

A critical insight from Milestone 1 is that **toxicity classification is fundamentally different from threat-intent classification**:

1. **What `unitary/toxic-bert` classifies**:
   - Vulgarity, profanity, and offensive adjectives (e.g. `obscene`, `insult`).
   - Generic hostility or coarse language.
   - It outputs an auxiliary `threat` label that flags blunt physical threat words ("I will kill you").
2. **What `unitary/toxic-bert` FAILS to classify**:
   - **Conditional Extortion**: *"If you don't send 500 dollars, I will forward your pictures to your employer."* (Notice: this sentence contains zero insults or curse words, so toxicity models often score it low, despite it being a severe crime).
   - **Coercive Control**: *"Don't you dare tell the police or talk to your family."*
   - **Urgency Manipulation**: *"You have 2 hours before the countdown ends."*

Therefore, SafeHER treats `unitary/toxic-bert` as an **Auxiliary Toxicity & Aggression Signal Provider**, running in tandem with our contextual threat intent engine rather than replacing it.

---

## 3. Hardware-Aware Analysis (AMD Ryzen 5 3450U, ~5.89 GB RAM)

### Target Environment Specifications
- **Processor**: AMD Ryzen 5 3450U (4 cores, 8 threads, 2.1 GHz base, up to 3.5 GHz boost).
- **Available System Memory**: Total ~5.89 GB (with ~1.0–2.0 GB active free headroom).
- **Execution Target**: CPU inference only; no dedicated high-memory GPU.
- **Python Runtime**: Python 3.12 (Windows x86-64).

### Key Architectural Guidelines
1. **Never Load Models Concurrently**: Loading BERT (500 MB), DeBERTa (350 MB), and an NLI model (250 MB) simultaneously alongside FastAPI and PostgreSQL risks paging or memory exhaustion.
2. **Lazy Initialization**: Load weights into RAM strictly upon the first inference call, not at application import time.
3. **Quantization & Torch Thread Capping**: Cap PyTorch CPU threads (`torch.set_num_threads(2)` or `4`) to prevent core saturation from degrading API responsiveness.

---

## 4. Recommended Multi-Stage Integration Roadmap

1. **Milestone 2 (Current)**:
   - Integrate **`unitary/toxic-bert`** with lazy loading, multilabel sigmoid scoring, and inference timing.
   - Map its outputs strictly to auxiliary signals (`toxic`, `severe_toxic`, `insult`, `identity_hate`) and use its `threat` signal as an input to augment the threat indicator pool.
   - Keep scores labeled as **uncalibrated raw sigmoid values**.
2. **Milestone 3 (Next)**:
   - Integrate lightweight zero-shot NLI (**`MoritzLaurer/DeBERTa-v3-xsmall-mnli`**) for conditional blackmail and exposure intent verification.
   - Implement temperature scaling calibration.
