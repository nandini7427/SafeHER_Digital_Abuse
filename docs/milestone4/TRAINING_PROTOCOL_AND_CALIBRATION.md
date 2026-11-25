# SafeHER M4 — Training Protocol and Calibration Plan

**Status:** Training framework implemented; full-scale model training and neural calibration not completed.\
**Last reviewed:** 25 November 2026

## Implemented engineering components

The repository includes a PyTorch training pipeline, model architectures, deterministic seeding/configuration, AdamW, focal-loss support, learning-rate scheduling, early stopping, checkpoint management, manifest/rights checks, safe checkpoint loading and calibration/uncertainty utilities.

Unit tests exercise the training and checkpoint mechanics on controlled fixture data. They do not constitute full-scale training or prove the model's detection capability.

## Training has not been completed

No approved trained Model A or Model B checkpoint and no populated rights-cleared M4 dataset were included in the supplied archive. Do not claim the architectures have been trained on FaceForensics++, GenImage or other external corpora unless the exact acquired dataset version, training manifest, logs, checkpoint and rights review are available.

## Required training record

For each model run, preserve:

- Dataset source/version, manifest SHA-256, split manifest SHA-256 and item counts by class/group.
- Exact code revision and dependency lock/versions.
- Architecture, initialization source and checkpoint SHA-256.
- Seed, device, batch size, optimizer, scheduler, loss, augmentations and preprocessing version.
- Epoch-by-epoch training and validation loss/metrics.
- Early-stopping decision and best checkpoint selection rule.
- Training completion metadata and dataset version list.
- Separate final test report on data not used for training, model selection, threshold selection or calibration.

## Calibration requirements

Fit calibration only after a trained checkpoint has been frozen, using a separate calibration split. Evaluate on a separate held-out test set. Report sample counts, class balance, calibration method, fitting procedure, reliability diagram, ECE definition/binning, Brier score, confidence intervals where appropriate, and performance under distribution shift. Small or unrepresentative calibration samples must be described as exploratory.

Earlier reports contain calibration outputs produced by controlled fixture/harness code. They are not evidence of calibration for a genuinely trained M4 neural detector and must not be quoted as Model A or Model B reliability.

## Safe activation gate

Operational inference requires verified completed-training metadata, architecture match, provenance hashes, dataset versions, artifact checksum and `COMMERCIAL_TRAINED` approval. If these conditions are not met, the model must remain unavailable; do not bypass the policy with `enforce_trained=False`.
