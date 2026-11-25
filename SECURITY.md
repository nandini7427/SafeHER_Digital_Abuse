# Security Policy

SafeHER is a hackathon/research prototype and is not approved for production handling of confidential survivor evidence.

## Reporting a vulnerability

Do not publish exploit details or real survivor data in a public issue. Contact the project maintainers through the private contact channel configured by the repository owner. If no private reporting channel has been configured, do not include sensitive details in a public issue; ask the repository owner to establish one.

## Sensitive-data rules

- Never commit environment files, tokens, passwords, private evidence, local databases, logs or raw datasets.
- Rotate any credentials that may have appeared in previously shared archives.
- Use generated or consent-cleared data for development and demonstrations.
- Keep survivor evidence out of all model training, calibration and benchmark pipelines.
- Use encrypted storage volumes and managed keys before storing confidential evidence; current application-level encryption at rest is not implemented.

## Model artifact rules

- Do not load untrusted pickle/joblib artifacts.
- Use SafeTensors where possible or PyTorch `weights_only=True` loading.
- Require artifact checksum and training provenance verification.
- Only artifacts with reviewed `COMMERCIAL_TRAINED` usage state may be considered for operational inference.
- Keep M4 disabled until properly trained and independently evaluated weights are approved.
