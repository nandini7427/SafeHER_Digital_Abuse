# SafeHER M4 Model Card — Current Status

**Status:** Research and engineering prototype; no approved trained neural model is included.\
**Last reviewed:** 25 November 2026

## Intended purpose

SafeHER M4 is designed to explore two separate research tasks:

- **Model A:** classify whether an image may contain AI-generated visual content.
- **Model B:** inspect a detected face region for patterns associated with supported facial-manipulation classes.

These are separate tasks. Neither model can establish who created an image, the identity of a depicted person, consent, intent, criminality, or legal authenticity.

## Current implementation

- Model A architectures include MobileNetV3-Large and EfficientNet-B0.
- Model B uses a patch-based facial-forensics architecture and a face-region preprocessing pipeline.
- Dataset manifests, provenance/rights validators, leakage checks, training utilities, artifact validation, uncertainty/calibration code and backend/frontend integration scaffolding exist.
- No approved full-scale trained Model A or Model B checkpoint is included in this repository.
- No populated, rights-cleared training dataset is included.
- Neural image inference is disabled by default in `.env.example`.

## Performance

**No validated Model A or Model B accuracy, precision, recall, F1, AUROC, calibration or real-world robustness figure is published by this model card.** Earlier reports contain fixture/harness measurements and unverified claims; they are superseded by this status document and must not be used as trained-model performance claims.

## Training data

No actual image dataset inventory or dataset payload was included in the supplied project ZIP. Dataset module code and test fixtures do not establish dataset acquisition or license permission. Before training, record the exact upstream source, dataset version, item-level provenance where required, license terms, attribution obligations, consent/usage restrictions, split assignments and manifest hashes.

Survivor-submitted evidence must never be used for model training, fine-tuning, calibration or benchmark construction.

## Limitations

- Untrained or unapproved weights are rejected by the inference loaders.
- Screenshots, heavy compression, crops, low-resolution faces, unfamiliar generators and unsupported domains may be inconclusive or unsupported.
- Absence of detected indicators does not establish authenticity.
- Model outputs are not legal or forensic proof.
- Human review and contextual corroboration are required for consequential decisions.

## Deployment restrictions

Do not enable M4 inference until approved trained checkpoints, provenance, separate validation/calibration/test splits, independently evaluated performance and rollback artifacts are available. Use the `COMMERCIAL_TRAINED` approval state only after documented rights and release review. The metadata field itself is not a legal opinion or substitute for source-license verification.
