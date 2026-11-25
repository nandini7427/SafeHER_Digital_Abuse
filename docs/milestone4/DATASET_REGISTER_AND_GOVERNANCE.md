# SafeHER M4 — Dataset Register and Governance

**Status:** Dataset infrastructure implemented; actual training datasets not supplied in the project archive.\
**Last reviewed:** 25 November 2026

## Current dataset status

SafeHER contains dataset schema, registry, provenance, integrity, duplicate detection, split/leakage, rights-validation, loading and label-adjudication modules. These modules provide controls for future datasets; they do not prove that any external dataset has been acquired, licensed, reviewed or ingested.

The supplied archive did not contain a populated image dataset manifest or dataset image corpus in the M4 dataset directory. Do not treat names of possible research datasets in older documents as proof that those datasets were used.

## Dataset admission requirements

For each proposed dataset, the owner must record:

- Exact dataset name, upstream URL, version/release date and acquisition date.
- License text and exact applicable terms; commercial and redistribution rights must be checked separately.
- Attribution, share-alike, non-commercial, research-only or other obligations.
- Data origin, subject/video/source grouping, collection method and known consent/usage restrictions.
- Sample count, class labels, annotation definitions and known label ambiguity.
- SHA-256, duplicate/near-duplicate groups and parent lineage where applicable.
- Train/validation/calibration/test/locked-test partition assignment.
- A documented decision approving or rejecting use for the intended model and deployment context.

A permissive-looking dataset landing page or repository license does not automatically grant rights to every image or model derivative. Seek qualified legal review for uncertain terms.

## Split and leakage controls

Use group-aware splits based on subject identity, source video, parent image, manipulation lineage and generator family as relevant. Keep calibration data separate from training and the final test set. Do not tune thresholds against the final locked test set. Record split manifest hashes and preserve an immutable experiment record.

## Survivor evidence separation

Survivor-submitted evidence is case material, not training data. It must not be copied into model datasets, used for automatic fine-tuning, or included in evaluation runs. The backend's private evidence directory and dataset directories must remain separate.

## Rights status labels

`RESEARCH_RESTRICTED`, `PENDING_REVIEW`, and `REJECTED` artifacts are not eligible for operational inference. The secure artifact loader now requires `COMMERCIAL_TRAINED` for operational use. That state is an internal approval gate, not proof of legal ownership or a substitute for license review.

## Current register

| Dataset | Status | Reason |
|---|---|---|
| Model A training data | Not registered in supplied archive | No populated manifest or image corpus supplied |
| Model B training data | Not registered in supplied archive | No populated manifest or image corpus supplied |
| Calibration data | Not available as a trained-model calibration set | No approved trained neural checkpoint/dataset lineage supplied |
| Locked test data | Not independently established | No external blind test custodian or frozen neural checkpoint evidenced |

Do not populate this table with estimated sample counts. Update it only when the dataset files, rights evidence and manifests are actually present and reviewed.
