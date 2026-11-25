# SafeHER — AI/ML Core Module

The `ai_core` module provides AI and machine learning capabilities for SafeHER: an AI-Powered Digital Abuse Intelligence & Evidence Response Platform.

## Module Structure

```text
ai_core/
│
├── __init__.py           # Package entry point
│
├── threat/               # Threat assessment & text abuse triage
│   ├── __init__.py
│   ├── schemas.py        # Pydantic v2 schemas and taxonomy
│   ├── preprocessing.py  # Text normalization and working-copy isolation
│   └── analyzer.py       # ThreatAnalyzer interface and baseline analyzer
│
├── evaluation/           # Benchmark protocols and dataset evaluation
│   ├── __init__.py
│   └── README.md
│
├── config.py             # AI/ML module settings & disclaimers
└── README.md             # Architecture and module documentation
```

---

## 1. Threat Taxonomy

SafeHER delineates distinct abuse intentions rather than collapsing all negative text into a single "toxicity" score. Toxicity reflects offensive language, whereas threat intent indicates coercion, violence, or extortion.

The 7 primary categories are:

| Category | Identifier | Description |
| :--- | :--- | :--- |
| **Direct Threats** | `direct_threat` | Statements expressing intent to inflict physical injury, bodily harm, violence, or death. |
| **Blackmail** | `blackmail` | Conditional demands coercing action under threat of revealing private, compromising, or fabricated material. |
| **Financial Extortion**| `financial_extortion`| Demands for monetary payment, cryptocurrency, gift cards, or financial assets under duress. |
| **Exposure Threats** | `exposure_threat` | Threats to leak, distribute, or publicize non-consensual intimate imagery (NCII), personal chats, or sensitive data (doxxing). |
| **Coercion** | `coercion` | Compelling involuntary behavior (e.g., demanding additional media, forbidding police contact, isolating victim). |
| **Harassment & Intimidation** | `harassment_intimidation` | Repeated, hostile, menacing, or degrading conduct aimed at inducing fear or distress. |
| **Urgency & Pressure** | `urgency_pressure` | Artificial temporal deadlines ("you have 1 hour", "countdown", "last chance") designed to induce panic and compliance. |

---

## 2. Working Copies & Evidence Integrity

In digital evidence workflows, original files and texts must **never** be altered.
- `ai_core.threat.preprocessing` accepts an input string and generates a `PreprocessedText` object containing a clean `working_copy`.
- The original string remains completely unaltered.
- Downstream NLP models operate exclusively on the `working_copy`.

---

## 3. Calibration & AI Disclaimers

Automated model outputs are subject to calibration rules:
1. **Uncalibrated Heuristics**: The development analyzer does not fabricate probabilistic confidence scores. It explicitly sets `is_calibrated = False`, providing qualitative threat levels rather than deceptive percentages.
2. **Mandatory Disclaimer**: Every response payload includes a mandatory legal and medical disclaimer emphasizing that SafeHER is an assistive triage tool and not forensic or legal proof.

---

## 4. Future Model Integration Roadmap

1. **Pretrained Transformer Integration**: Fine-tuned sequence classification models (e.g. `unitary/toxic-bert` for auxiliary toxicity, plus domain-adapted extortion/threat classifiers).
2. **Span Extraction**: Token classification (NER-style sequence labelling) for precise explainable text-span highlighting.
3. **Multilingual Expansion**: Support for Indic languages (Hindi, Kannada, Tamil, etc.) via IndicBERT / multilingual embeddings.
