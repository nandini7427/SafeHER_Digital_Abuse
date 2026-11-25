# SafeHER — Local Development and Deployment Notes

**Status:** Local prototype instructions; production deployment is not approved.\
**Last reviewed:** 25 November 2026

## Supported development approach

Use a local Python environment, SQLite for development only, and Vite for the frontend. Production requires a managed database, HTTPS, strong environment-managed secrets, restricted filesystem permissions, operational monitoring, backup/deletion procedures and security review.

## Evidence storage and encryption

The current evidence service writes original evidence bytes into the configured private storage directory using a temporary-file/atomic-replace pattern and records SHA-256 integrity metadata. It enforces ownership/access checks and privacy-safe logging.

**The supplied application does not implement application-level encryption at rest or managed encryption-key handling.** A private path and SHA-256 hash are not encryption. Before any real confidential evidence is stored, use an encrypted storage volume/service with managed keys, least-privilege access, backup encryption and tested deletion/recovery procedures. Do not use real intimate images in an unreviewed hackathon deployment.

## Neural model activation

M4 neural inference is disabled by default. Keep it disabled until a genuinely trained model has approved dataset provenance, training completion metadata, verified checksum, independent validation and a rollback artifact. The loader rejects missing or incomplete training provenance and requires the approved operational usage state.

## Frontend setup

From `frontend/` run:

```powershell
npm ci
npm test -- --run
npm run build
```

Do not copy `node_modules` between Windows and Linux. Reinstall using the lockfile on the target OS.

## Production checklist

- `ENVIRONMENT=production`, `DEBUG=false`, strong unique `JWT_SECRET_KEY` (32+ characters).
- Managed PostgreSQL or equivalent; no SQLite.
- TLS termination and secure headers.
- Encrypted persistent storage and managed keys for evidence.
- Private storage not mounted into public web roots.
- Backups encrypted and restore-tested; retention/deletion policy documented.
- No `.env`, private evidence, databases, local datasets or model binaries in Git.
- Rate limits, upload quotas, monitoring, alerting and incident-response plan.
- Verified model artifacts only; M4 disabled until approved trained checkpoints exist.
- Linux/container integration tests and capacity tests completed on deployment hardware.
- Independent security/privacy review completed.
