# SafeHER Hackathon Demo Guide

## Demo objective

Show an end-to-end survivor-centred workflow: create a private incident case, assess a threatening message, analyze a consent-cleared sample image using the available M3 technical heuristics, preserve an evidence integrity record, and explain the human-review path. Keep the demo honest about M4: neural detectors are not trained/approved in this repository and remain disabled by default.

## Before the demo

1. Use a clean local environment and run database migrations.
2. Start the API and frontend using the root README instructions.
3. Create a demo user with a strong password.
4. Use only generated, public-domain or consent-cleared sample content. Never use real intimate images or real survivor information in a hackathon environment.
5. Confirm that M4 is disabled (`AI_ENABLE_NEURAL_DETECTOR=false`).
6. Have one consent-cleared ordinary image and, if using the neural panel, explain that the current live pipeline does not include a face-localization detector. Model B reports `FACE_LOCALIZATION_UNAVAILABLE` unless verified face boxes or reviewed annotations are supplied. Explain that ELA/FFT are heuristic indicators, not deepfake proof.

## Suggested 4–5 minute walkthrough

### 1. Introduce the problem

Explain that people facing digital abuse may need a safe way to organize messages and image evidence, understand technical findings, and request human help. Avoid saying that AI can prove an image is fake or determine consent or identity.

### 2. Create an incident case

Sign in, open **Incident cases**, create a demo case and show that the case list is tied to the authenticated account.

### 3. Assess a threatening message

Open **Message assessment** and paste a synthetic example of coercion or image-based blackmail. Show the detected wording, explanation, uncertainty and mandatory disclaimer. Emphasize that the system flags language patterns and does not make a legal finding.

### 4. Analyze a sample image

Open **Image evidence**, associate the analysis with the demo case if needed, and upload a consent-cleared sample. Explain the available image-quality, ELA/FFT and evidence-integrity signals. If the neural panel reports unavailable or face localization unavailable, state clearly that trained M4 weights and a verified face-localization component are still research deliverables and are not enabled in this demo.

### 5. Close with the trust roadmap

Show the audit status document and the next gates: rights-cleared datasets, full training runs, locked test splits, calibrated evaluation, independent review, and encrypted storage before any real confidential evidence is handled.

## Questions judges may ask

**Does SafeHER detect every deepfake?**\
No. The M4 architecture is implemented, but approved trained detector weights and scientific validation are not yet complete.

**Is the AI result legal proof?**\
No. All image analysis is triage support only; identity, consent, intent and criminality cannot be inferred from pixels alone.

**Are uploaded images encrypted?**\
The current prototype stores evidence in a private directory and records SHA-256 integrity metadata. Application-level encryption at rest is not implemented; real confidential evidence must not be used until encrypted storage and key management are deployed.

**Are survivor uploads used to train models?**\
No. Case evidence must remain separate from training datasets and evaluation corpora.

**What is the next technical milestone?**\
Acquire and clear datasets, train Model A and Model B, preserve complete provenance, evaluate on held-out data and undergo independent review.
