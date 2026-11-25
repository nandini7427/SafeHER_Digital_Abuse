# SafeHER

**Detect. Preserve. Protect. Empower.**

SafeHER is a survivor-centred digital safety prototype for organizing incident information, assessing threatening messages, preserving image evidence with integrity metadata, and supporting human review. It is being developed as a hackathon project; it is **not a certified forensic product, legal service, or production-ready service for confidential survivor evidence**.

## Current milestone status

| Milestone | Scope | Current status |
|---|---|---|
| M1 | Threat and coercive-message assessment | Pattern-based assessment and optional Toxic-BERT integration are implemented. Transformer inference is disabled by default until a separately verified local model artifact is configured. |
| M2 | API, authentication and incident cases | FastAPI authentication, owner-scoped case APIs, and frontend sign-in/case workspace are implemented. |
| M3 | Image intake, evidence integrity, ELA/FFT and review | Private evidence storage, SHA-256 integrity checks, image-quality/ELA/FFT analysis and human-review workflow are implemented. Application-level encryption at rest is **not** implemented; configure encrypted storage at the deployment layer before handling confidential material. |
| M4 | Neural AI-generated-image and facial-manipulation intelligence | Model architectures, dataset tooling, training framework, artifact handling and integration scaffolding are implemented. **No approved, full-scale trained Model A or Model B checkpoint is included in this repository. No face-localization detector is shipped or wired into the live Model B path; without verified face boxes, Model B correctly reports `FACE_LOCALIZATION_UNAVAILABLE`. Neural inference is disabled by default. Do not present M4 as a validated detector.** |

See [the current audit and release status](docs/PROJECT_STATUS_AND_VERIFICATION.md) for verified test results, limitations and next steps.

## What SafeHER does not claim

- An AI score is not proof that an image is fake or authentic.
- Image analysis cannot establish identity, consent, intent or criminality.
- A negative or unavailable analysis does not show that abuse did not occur.
- The project does not provide legal advice or a forensic certification.
- Uploaded survivor evidence is not training data. Never add private evidence to any dataset or benchmark.

## Repository layout

```text
ai_core/             Threat, image-intelligence, dataset and neural code
backend/             FastAPI application, services, migrations and tests
frontend/            React + TypeScript + Vite interface
tests/               AI and core Python tests
docs/                Architecture, milestone reports and release status
requirements-ai.txt  AI/ML dependencies
```

## Local development setup

Tested commands below assume Windows PowerShell from the repository root. Use a clean Python virtual environment and install dependencies in the order shown.

### 1. Backend environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install -r requirements-ai.txt
Copy-Item .env.example .env
```

The example configuration uses a local SQLite development database. Blank `JWT_SECRET_KEY` generates an ephemeral process-local secret only in `development` and `test`; `staging` and `production` require an explicitly configured non-placeholder secret of at least 32 characters (`production` also rejects SQLite and `DEBUG=true`); see `backend/app/config.py` and the deployment notes.

### 2. Database migration

```powershell
python -m alembic -c alembic.ini upgrade head
```

### 3. Start the API

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### 4. Start the frontend (new PowerShell window)

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open the local URL printed by Vite. The Vite development proxy routes `/api/v1` to the backend at `http://127.0.0.1:8000`.

### 5. Run tests

From the repository root:

```powershell
python -m pytest tests\ai -q
python -m pytest backend\tests -q
```

From `frontend`:

```powershell
npm ci
npm test -- --run
npm run build
```

A test count is not a quality guarantee. Review the current audit report for the exact verification performed on this package and what remains unverified.

## M4 model activation policy

M4 neural inference is off by default. Keep it disabled unless both model artifacts have:

1. Completed training on documented, appropriately licensed data.
2. Provenance metadata and verified hashes.
3. An independently reviewed train/validation/test split with no leakage.
4. Separate held-out performance and calibration reports.
5. Explicit operational approval and a rollback plan.

Only after those requirements are met should `AI_ENABLE_NEURAL_DETECTOR` and the relevant model paths/hashes be configured. Do not use `enforce_trained=False` as a production workaround. The current loaders reject incomplete training provenance.

## Security and privacy before any demo deployment

- Keep `.env`, databases, model weights, private evidence, logs and datasets out of Git.
- The uploaded source archive originally contained an `.env`; rotate any credentials or secrets that were ever used in it before publishing the repository.
- Do not upload real intimate images or private survivor material during a hackathon demo. Use consent-cleared or appropriately licensed test material.
- Private evidence storage currently relies on filesystem privacy controls and integrity hashes, not application-level encryption. Use encrypted disks/volumes and managed key protection before handling sensitive data.
- Use HTTPS, a production JWT secret, a managed database, restrictive filesystem permissions and tested backup/deletion policies in any hosted deployment.

## GitHub contribution / release workflow

Canonical GitHub repository: `https://github.com/nandini7427/SafeHER_Digital_Abuse.git` (Milestone 4 feature branch: `feature/milestone-4-neural-deepfake`).

Before pushing, inspect the source tree and run:

```powershell
git status --short
git diff --check
git check-ignore -v .env frontend/node_modules frontend/dist private_evidence
```

Review all files before `git add`. Do not use `git add .` until you have confirmed no secrets, private evidence, model weights, generated dependencies or runtime data are present. Keep model datasets and artifacts in an approved storage location and publish hashes/manifests rather than the image data itself.

## License and attribution

Add the project's chosen open-source license and third-party dataset/model attributions only after checking contributor ownership and source terms. Dataset code and a license-validation utility do not themselves grant rights to external images or pretrained weights.
