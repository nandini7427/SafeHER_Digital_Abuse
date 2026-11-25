# Contributing to SafeHER

Thank you for helping improve SafeHER. This repository handles potentially sensitive digital-abuse workflows, so correctness, privacy and honest claims are more important than adding features quickly.

## Development principles

- Preserve existing milestone behavior and database migrations.
- Do not commit `.env`, credentials, local databases, private evidence, uploaded files, logs, raw datasets or trained model binaries.
- Never use survivor evidence for training, fine-tuning, calibration or testing.
- Do not fabricate AI scores, heatmaps, benchmark data or training results.
- Do not describe model output as proof of identity, consent, criminality, authenticity or legal findings.
- Keep neural inference disabled until trained artifacts pass documented data-rights, provenance, evaluation and release gates.
- Use additive, reviewed schema migrations; do not edit applied migrations.

## Before opening a pull request

```powershell
python -m compileall -q ai_core backend tests
python -m pytest tests\ai -q
python -m pytest backend\tests -q
```

From `frontend/`:

```powershell
npm ci
npm test -- --run
npm run build
```

Also review:

```powershell
git status --short
git diff --check
git check-ignore -v .env frontend/node_modules frontend/dist private_evidence
```

## Model and dataset changes

Every model change must include an exact architecture/version, source-code revision, training manifest hash, split manifest hash, dataset versions, training logs, artifact SHA-256, evaluation protocol and limitations. Dataset licenses and usage restrictions must be reviewed for the intended use. Test data must remain separate from training and calibration data.
