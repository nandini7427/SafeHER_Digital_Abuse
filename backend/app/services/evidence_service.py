"""Evidence management, cryptographic integrity verification, and secure lifecycle service.

Enforces:
- Original evidence preservation and SHA-256 cryptographic immutability.
- Strict horizontal authorization and case ownership validation.
- Path traversal prevention and private storage isolation.
- File integrity verification against disk tampering.
- Controlled evidence deletion with physical cleanup and audit logging.
- Privacy-safe logging (never leaks raw image bytes or base64 data).
"""

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
from pathlib import Path
import time
from typing import List, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models import AuditLog, Case, Evidence, EvidenceCleanupLease, ImageReview, User
from backend.app.schemas import EvidenceIntegrityResponse, EvidenceResponse

logger = logging.getLogger("safeher.backend.services.evidence")


def _canonical_storage_key(filename: str) -> str:
    """Map a staging filename back to the Evidence storage key.

    Evidence files are named ``<uuid>.bin`` and staging files are created with
    ``Path.with_suffix('.part')`` as ``<uuid>.part``. Removing only ``.part``
    would incorrectly produce ``<uuid>`` and fail to match the upload lease.
    """
    if filename.endswith(".part"):
        return f"{filename[:-5]}.bin"
    return filename


def _storage_path_candidates(directory: Path, storage_key: str) -> list[Path]:
    """Return canonical and staging paths for an Evidence storage key."""
    candidates = [directory / storage_key]
    if storage_key.endswith(".bin"):
        candidates.append(directory / f"{storage_key[:-4]}.part")
    return candidates


class EvidenceService:
    """Service layer managing evidence retrieval, integrity audits, and safe deletion."""

    def _get_authorized_evidence(
        self,
        evidence_id: UUID,
        current_user: User,
        db: Session,
        allow_reviewer: bool = False,
    ) -> Evidence:
        """Query evidence by ID and enforce strict horizontal ownership or reviewer access.

        Raises:
            HTTPException: 404 if evidence not found.
            HTTPException: 403 if evidence belongs to another user and caller is not an authorized reviewer.
        """
        evidence = db.scalar(select(Evidence).where(Evidence.id == evidence_id))
        if evidence is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence record not found.",
            )

        if evidence.owner_id == current_user.id:
            # Also verify associated case ownership
            case = db.scalar(select(Case).where(Case.id == evidence.case_id))
            if case is not None and case.owner_id != current_user.id:
                logger.warning(
                    "Cross-case boundary violation: user_id=%s, case_id=%s, evidence_id=%s",
                    current_user.id,
                    evidence.case_id,
                    evidence_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access evidence associated with this case.",
                )
            return evidence

        # Reviewer authorization: reviewer/advocate/admin inspecting evidence under review
        if allow_reviewer and current_user.role in ("reviewer", "advocate", "admin"):
            review_exists = db.scalar(
                select(ImageReview).where(ImageReview.evidence_id == evidence_id)
            )
            if review_exists is not None or current_user.role == "admin":
                return evidence

        logger.warning(
            "Unauthorized evidence access attempt: user_id=%s, evidence_id=%s, actual_owner=%s",
            current_user.id,
            evidence_id,
            evidence.owner_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this evidence.",
        )

    def _resolve_safe_path(self, storage_key: str) -> Path:
        """Resolve storage key within PRIVATE_STORAGE_PATH, preventing path traversal.

        Raises:
            HTTPException: 400 if path traversal or invalid destination detected.
        """
        storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
        if (
            not storage_key
            or Path(storage_key).name != storage_key
            or "/" in storage_key
            or "\\" in storage_key
        ):
            logger.critical("Path traversal attempt intercepted: storage_key=%s", storage_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid evidence storage destination.",
            )

        dest_path = (storage_dir / storage_key).resolve()
        try:
            dest_path.relative_to(storage_dir)
        except ValueError as exc:
            logger.critical("Path traversal attempt intercepted: storage_key=%s", storage_key)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid evidence storage destination.",
            ) from exc

        return dest_path

    def get_evidence_metadata(
        self,
        evidence_id: UUID,
        current_user: User,
        db: Session,
    ) -> EvidenceResponse:
        """Retrieve sanitized metadata for an owned evidence record.

        Privacy guarantee: Does NOT expose internal storage keys or filesystem paths.
        """
        evidence = self._get_authorized_evidence(evidence_id, current_user, db)

        # Record audit log
        audit = AuditLog(
            user_id=current_user.id,
            action="EVIDENCE_METADATA_ACCESSED",
            entity_type="evidence",
            entity_id=str(evidence.id),
        )
        db.add(audit)
        db.commit()

        logger.info(
            "Evidence metadata retrieved: user_id=%s, evidence_id=%s, sha256_prefix=%s",
            current_user.id,
            evidence.id,
            evidence.sha256[:8],
        )

        return EvidenceResponse.model_validate(evidence)

    def verify_integrity(
        self,
        evidence_id: UUID,
        current_user: User,
        db: Session,
    ) -> EvidenceIntegrityResponse:
        """Verify the cryptographic SHA-256 integrity of stored evidence against disk.

        Detects file modification, bitrot, or unauthorized tampering.
        """
        evidence = self._get_authorized_evidence(evidence_id, current_user, db)
        dest_path = self._resolve_safe_path(evidence.storage_key)

        now = datetime.now(timezone.utc)

        if not dest_path.exists():
            logger.error("Evidence file missing from private storage: evidence_id=%s", evidence_id)
            return EvidenceIntegrityResponse(
                evidence_id=evidence.id,
                sha256=evidence.sha256,
                is_valid=False,
                verified_at=now,
                message="CRITICAL: Evidence file missing from private storage vault.",
            )

        try:
            file_bytes = dest_path.read_bytes()
            computed_sha256 = hashlib.sha256(file_bytes).hexdigest()
        except Exception as exc:
            logger.error("Failed to read evidence file for integrity audit: %s", str(exc))
            return EvidenceIntegrityResponse(
                evidence_id=evidence.id,
                sha256=evidence.sha256,
                is_valid=False,
                verified_at=now,
                message="Error reading evidence file from secure storage vault.",
            )

        is_valid = (computed_sha256 == evidence.sha256)

        if is_valid:
            message = "Evidence integrity verified: SHA-256 matches pristine baseline."
            logger.info("Evidence integrity verified: evidence_id=%s", evidence_id)
        else:
            message = "CRITICAL: Evidence hash mismatch. File content has been modified or corrupted."
            logger.critical(
                "Evidence tampering detected: evidence_id=%s, expected_sha=%s, computed_sha=%s",
                evidence_id,
                evidence.sha256,
                computed_sha256,
            )

        # Record audit log
        audit = AuditLog(
            user_id=current_user.id,
            action="EVIDENCE_INTEGRITY_VERIFIED",
            entity_type="evidence",
            entity_id=str(evidence.id),
        )
        db.add(audit)
        db.commit()

        return EvidenceIntegrityResponse(
            evidence_id=evidence.id,
            sha256=evidence.sha256,
            is_valid=is_valid,
            verified_at=now,
            message=message,
        )

    def retrieve_evidence_stream(
        self,
        evidence_id: UUID,
        current_user: User,
        db: Session,
    ) -> Tuple[Path, str, str]:
        """Authorize caller and return safe physical path, mime type, and sanitized filename.

        Returns:
            Tuple of (physical_path, mime_type, original_filename).
        """
        evidence = self._get_authorized_evidence(evidence_id, current_user, db, allow_reviewer=True)
        dest_path = self._resolve_safe_path(evidence.storage_key)

        if not dest_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence file not found in storage vault.",
            )

        # Distinguish standard owner download vs. authorized human review inspection
        audit_action = (
            "HUMAN_REVIEW_EVIDENCE_ACCESSED"
            if evidence.owner_id != current_user.id
            else "EVIDENCE_DOWNLOADED"
        )
        audit = AuditLog(
            user_id=current_user.id,
            action=audit_action,
            entity_type="evidence",
            entity_id=str(evidence.id),
        )
        db.add(audit)
        db.commit()

        logger.info(
            "Evidence stream authorized: user_id=%s, evidence_id=%s, action=%s",
            current_user.id,
            evidence_id,
            audit_action,
        )
        return dest_path, evidence.mime_type, evidence.original_filename

    def delete_evidence(
        self,
        evidence_id: UUID,
        current_user: User,
        db: Session,
    ) -> None:
        """Securely remove evidence record and wipe file from disk under controlled authorization."""
        evidence = self._get_authorized_evidence(evidence_id, current_user, db)
        storage_key = evidence.storage_key
        dest_path = self._resolve_safe_path(storage_key)

        # Commit the audit event and database deletion before removing the file.
        # If the database commit fails, the original evidence file remains available.
        audit = AuditLog(
            user_id=current_user.id,
            action="EVIDENCE_DELETED",
            entity_type="evidence",
            entity_id=str(evidence.id),
        )
        db.add(audit)
        db.delete(evidence)
        db.commit()

        # If unlink fails, the now-unreferenced file is a safe orphan and can be
        # recovered/reconciled by the administrative cleanup process.
        try:
            dest_path.unlink(missing_ok=True)
            logger.info("Physical evidence file removed: %s", storage_key)
        except OSError as exc:
            logger.error(
                "Evidence DB record deleted but physical cleanup is pending: storage_key=%s, error=%s",
                storage_key,
                str(exc),
            )

        logger.info("Evidence record deleted: user_id=%s, evidence_id=%s", current_user.id, evidence_id)

    def restore_from_quarantine(self, db: Session, filename: str) -> bool:
        """Safely restore a quarantined file back to canonical storage.

        The filename is restricted to a single vault filename. If the database
        commit fails after the atomic move, the file is moved back into quarantine
        so the database state and filesystem location remain aligned.
        """
        if (
            not filename
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or not (filename.endswith(".bin") or filename.endswith(".part"))
        ):
            logger.error("Invalid quarantine filename rejected: %s", filename)
            return False

        storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
        quarantine_dir = (storage_dir / "quarantine").resolve()
        source_path = (quarantine_dir / filename).resolve()
        dest_path = (storage_dir / filename).resolve()
        try:
            source_path.relative_to(quarantine_dir)
            dest_path.relative_to(storage_dir)
        except ValueError:
            logger.error("Quarantine path escaped storage directory: %s", filename)
            return False

        if not source_path.exists() or not source_path.is_file() or dest_path.exists():
            logger.error("Quarantined file missing or destination already exists: %s", filename)
            return False

        moved = False
        try:
            source_path.replace(dest_path)
            moved = True

            canonical_key = _canonical_storage_key(filename)
            leases = db.scalars(
                select(EvidenceCleanupLease)
                .where(
                    EvidenceCleanupLease.storage_key == canonical_key,
                    EvidenceCleanupLease.status == "QUARANTINED"
                )
            ).all()
            now_dt = datetime.now(timezone.utc)
            for lease in leases:
                lease.status = "RESTORED"
                lease.resolved_at = now_dt
                lease.notes = "Restored from quarantine to canonical storage"
            db.commit()
            logger.info("Successfully restored file from quarantine: %s", filename)
            return True
        except Exception as exc:
            db.rollback()
            if moved and dest_path.exists():
                try:
                    dest_path.replace(source_path)
                except OSError as rollback_exc:
                    logger.critical(
                        "Database restore failed and filesystem compensation also failed: filename=%s, error=%s",
                        filename,
                        str(rollback_exc),
                    )
            logger.error("Failed restoring %s from quarantine: %s", filename, str(exc))
            return False

    def reconcile_orphan_evidence(
        self,
        db: Session,
        dry_run: bool = True,
        grace_period_seconds: int = 3600,
        quarantine: bool = False,
        quarantine_retention_seconds: int = 86400,
    ) -> List[str]:
        """Scan private storage for orphaned files not tracked in the Evidence database.

        Safety invariants:
        - Never deletes valid evidence keys referenced in the database.
        - Enforces a grace period (default 1 hour) so active concurrent uploads are never touched.
        - Database-backed cleanup leases: Before touching disk, registers an EvidenceCleanupLease record.
        - Crash recovery: Reconciles interrupted or stale CLAIMED leases from previous restarts.
        - Re-checks the database immediately before deletion/quarantine to guard against TOCTOU races.
        - Supports quarantine-and-delayed-deletion staging leases and recovery.
        - Defaults to dry_run=True to safely inspect orphans before physical modification.
        """
        storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
        if not storage_dir.exists():
            return []

        quarantine_dir = storage_dir / "quarantine"
        now = time.time()
        now_dt = datetime.now(timezone.utc)
        orphans: List[str] = []

        # 0. Crash recovery: Audit stale/interrupted CLAIMED leases from previous restarts
        if not dry_run:
            try:
                stale_claims = db.scalars(
                    select(EvidenceCleanupLease)
                    .where(EvidenceCleanupLease.status.in_(["CLAIMED", "PURGE_CLAIMED"]))
                ).all()
                for claim in stale_claims:
                    canonical_files = _storage_path_candidates(storage_dir, claim.storage_key)
                    quarantine_files = _storage_path_candidates(quarantine_dir, claim.storage_key)
                    quarantined_file = next((candidate for candidate in quarantine_files if candidate.exists()), None)
                    canonical_file = next((candidate for candidate in canonical_files if candidate.exists()), None)

                    if claim.status == "PURGE_CLAIMED":
                        # A quarantine-retention purge may have crashed between its
                        # intent commit and unlink. Keep it quarantined for retry if
                        # the file is still present; otherwise finalize the purge.
                        if quarantined_file is not None:
                            claim.status = "QUARANTINED"
                            claim.notes = "Recovered interrupted quarantine purge; file remains available for retention retry"
                        else:
                            claim.status = "PURGED"
                            claim.resolved_at = now_dt
                            claim.notes = "Recovered interrupted quarantine purge; file no longer exists"
                    elif quarantined_file is not None:
                        claim.status = "QUARANTINED"
                        claim.resolved_at = now_dt
                        claim.notes = "Recovered after crash: file verified in quarantine"
                    elif canonical_file is None:
                        claim.status = "PURGED"
                        claim.resolved_at = now_dt
                        claim.notes = "Recovered after crash: canonical and staging files are absent"
                    else:
                        is_linked = db.scalar(select(Evidence).where(Evidence.storage_key == claim.storage_key))
                        if is_linked is not None:
                            claim.status = "ABORTED"
                            claim.resolved_at = now_dt
                            claim.notes = "Aborted claim after crash: key now linked to Evidence"
                        else:
                            # The crash occurred after the claim commit but before
                            # the filesystem action. Release this stale claim so a
                            # later scan can safely acquire a fresh one.
                            claim.status = "ABORTED"
                            claim.resolved_at = now_dt
                            claim.notes = "Released interrupted cleanup claim; canonical/staging file remains eligible for re-scan"
                db.commit()
            except Exception as crash_exc:
                db.rollback()
                logger.error("Failed auditing stale cleanup leases: %s", str(crash_exc))

        # 1. Fetch all valid storage keys from database
        valid_keys = set(db.scalars(select(Evidence.storage_key)).all())

        # 2. Iterate through storage directory files safely
        for entry in storage_dir.iterdir():
            if entry.name == "quarantine":
                continue

            # Reconcile completed .bin files as well as abandoned temporary .part files
            if not entry.is_file() or (not entry.name.endswith(".bin") and not entry.name.endswith(".part")):
                continue

            # Strip .part suffix to check underlying storage key
            canonical_key = _canonical_storage_key(entry.name)

            # Check if file is referenced in database
            if canonical_key not in valid_keys:
                # Guard against active uploads: check if file has an active upload lease
                active_upload = db.scalar(
                    select(EvidenceCleanupLease).where(
                        EvidenceCleanupLease.storage_key == canonical_key,
                        EvidenceCleanupLease.status == "ACTIVE_UPLOAD",
                        EvidenceCleanupLease.lease_expires_at > now_dt,
                    )
                )
                if active_upload is not None:
                    logger.info(
                        "Skipping active upload during orphan scan: storage_key=%s, expires_at=%s",
                        canonical_key,
                        active_upload.lease_expires_at,
                    )
                    continue

                try:
                    mtime = entry.stat().st_mtime
                    age_seconds = now - mtime
                    # Only reconcile files older than the grace period
                    if age_seconds >= grace_period_seconds:
                        orphans.append(entry.name)
                        if dry_run:
                            logger.info(
                                "Proposed orphan reconciliation (dry-run): filename=%s, age_seconds=%.1f, action=%s",
                                entry.name,
                                age_seconds,
                                "quarantine" if quarantine else "purge",
                            )
                        else:
                            # Recheck DB immediately before claim to guard against TOCTOU race condition
                            recheck = db.scalar(select(Evidence).where(Evidence.storage_key == canonical_key))
                            recheck_upload = db.scalar(
                                select(EvidenceCleanupLease).where(
                                    EvidenceCleanupLease.storage_key == canonical_key,
                                    EvidenceCleanupLease.status.in_(["ACTIVE_UPLOAD", "COMPLETED"]),
                                    EvidenceCleanupLease.lease_expires_at > now_dt,
                                )
                            )
                            existing_cleanup_claim = db.scalar(
                                select(EvidenceCleanupLease).where(
                                    EvidenceCleanupLease.storage_key == canonical_key,
                                    EvidenceCleanupLease.status.in_(["CLAIMED", "PURGE_CLAIMED"]),
                                )
                            )
                            if recheck is not None or recheck_upload is not None or existing_cleanup_claim is not None:
                                logger.warning(
                                    "Race condition intercepted: %s was linked to database or active upload during scan. Skipping action.",
                                    canonical_key,
                                )
                                continue

                            # Step A: Register database cleanup claim/lease
                            lease = EvidenceCleanupLease(
                                storage_key=canonical_key,
                                status="CLAIMED",
                                action="quarantine" if quarantine else "purge",
                                lease_expires_at=now_dt + timedelta(seconds=quarantine_retention_seconds),
                            )
                            db.add(lease)
                            db.commit()

                            # Step B: Execute filesystem modification
                            try:
                                if quarantine:
                                    quarantine_dir.mkdir(parents=True, exist_ok=True)
                                    target_quarantine_path = quarantine_dir / entry.name
                                    entry.replace(target_quarantine_path)
                                    os.utime(target_quarantine_path, (now, now))
                                    lease.status = "QUARANTINED"
                                    lease.resolved_at = datetime.now(timezone.utc)
                                    db.commit()
                                    logger.warning(
                                        "Orphaned storage file moved to quarantine lease: filename=%s, age_seconds=%.1f",
                                        entry.name,
                                        age_seconds,
                                    )
                                else:
                                    entry.unlink(missing_ok=True)
                                    lease.status = "PURGED"
                                    lease.resolved_at = datetime.now(timezone.utc)
                                    db.commit()
                                    logger.warning(
                                        "Orphaned storage file purged: filename=%s, age_seconds=%.1f",
                                        entry.name,
                                        age_seconds,
                                    )
                            except Exception as fs_exc:
                                db.rollback()
                                # If the quarantine move succeeded but the status
                                # commit failed, compensate by moving the file back.
                                # If compensation also fails, leave the persisted
                                # CLAIMED lease for the next reconciliation recovery.
                                target_quarantine_path = quarantine_dir / entry.name
                                if quarantine and target_quarantine_path.exists() and not entry.exists():
                                    try:
                                        target_quarantine_path.replace(entry)
                                    except OSError as compensation_exc:
                                        logger.critical(
                                            "Cleanup state commit failed and quarantine compensation failed; lease left CLAIMED for recovery: filename=%s, error=%s",
                                            entry.name,
                                            str(compensation_exc),
                                        )
                                    else:
                                        try:
                                            persisted_lease = db.get(EvidenceCleanupLease, lease.id)
                                            if persisted_lease is not None:
                                                persisted_lease.status = "ABORTED"
                                                persisted_lease.resolved_at = datetime.now(timezone.utc)
                                                persisted_lease.notes = f"Quarantine move compensated after DB failure: {str(fs_exc)[:300]}"
                                                db.commit()
                                        except Exception:
                                            db.rollback()
                                elif entry.exists():
                                    # Filesystem operation did not take effect; mark
                                    # the claim aborted when the DB is available.
                                    try:
                                        persisted_lease = db.get(EvidenceCleanupLease, lease.id)
                                        if persisted_lease is not None:
                                            persisted_lease.status = "ABORTED"
                                            persisted_lease.resolved_at = datetime.now(timezone.utc)
                                            persisted_lease.notes = f"Filesystem operation failed: {str(fs_exc)[:350]}"
                                            db.commit()
                                    except Exception:
                                        db.rollback()
                                # When a purge already removed the file, or a move
                                # could not be compensated, keep CLAIMED for recovery.
                                logger.error("Failed filesystem/state operation for orphan %s: %s", entry.name, str(fs_exc))
                                continue
                except Exception as exc:
                    logger.error("Failed inspecting potential orphan %s: %s", entry.name, str(exc))

        # 3. Purge aged quarantine lease files older than quarantine_retention_seconds
        if not dry_run and quarantine_dir.exists():
            for q_entry in quarantine_dir.iterdir():
                if q_entry.is_file():
                    try:
                        q_age = now - q_entry.stat().st_mtime
                        if q_age >= quarantine_retention_seconds:
                            canonical_key = _canonical_storage_key(q_entry.name)
                            # Persist purge intent before touching disk. A crash after
                            # this commit is recoverable from PURGE_CLAIMED on the
                            # next reconciliation run.
                            leases = db.scalars(
                                select(EvidenceCleanupLease)
                                .where(
                                    EvidenceCleanupLease.storage_key == canonical_key,
                                    EvidenceCleanupLease.status == "QUARANTINED"
                                )
                            ).all()
                            for lease in leases:
                                lease.status = "PURGE_CLAIMED"
                                lease.notes = "Quarantine retention expired; permanent purge claimed"
                            db.commit()
                            q_entry.unlink(missing_ok=True)
                            purge_dt = datetime.now(timezone.utc)
                            for lease in leases:
                                lease.status = "PURGED"
                                lease.resolved_at = purge_dt
                                lease.notes = f"Quarantine retention expired ({quarantine_retention_seconds}s)"
                            db.commit()
                            logger.warning(
                                "Quarantined orphan lease expired and permanently purged: filename=%s, age=%.1fs",
                                q_entry.name,
                                q_age,
                            )
                    except Exception as q_exc:
                        db.rollback()
                        logger.error("Failed purging expired quarantine file %s: %s", q_entry.name, str(q_exc))

        return orphans


evidence_service = EvidenceService()


if __name__ == "__main__":
    import argparse
    import sys
    from backend.app.database import SessionLocal

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SafeHER Evidence Vault Orphan Reconciliation Tool")
    parser.add_argument("--reconcile", action="store_true", help="Run orphan reconciliation audit")
    parser.add_argument("--restore", type=str, default=None, help="Restore a specific file from quarantine back to vault")
    parser.add_argument("--no-dry-run", action="store_true", help="Execute active modification (defaults to dry-run audit)")
    parser.add_argument("--quarantine", action="store_true", help="Move orphans to quarantine/ lease rather than immediate deletion")
    parser.add_argument("--grace-period", type=int, default=3600, help="Grace period in seconds (default 3600)")
    parser.add_argument("--quarantine-retention", type=int, default=86400, help="Quarantine lease retention in seconds (default 86400)")

    args = parser.parse_args()

    if args.restore:
        with SessionLocal() as db_session:
            success = evidence_service.restore_from_quarantine(db=db_session, filename=args.restore)
            if success:
                print(f"Successfully restored {args.restore} from quarantine.")
                sys.exit(0)
            else:
                print(f"Failed to restore {args.restore} from quarantine.")
                sys.exit(1)

    if args.reconcile:
        is_dry_run = not args.no_dry_run
        mode_str = ("ACTIVE QUARANTINE LEASE" if args.quarantine else "ACTIVE DIRECT DELETION") if not is_dry_run else "DRY-RUN (AUDIT ONLY)"
        print(f"Executing SafeHER evidence vault orphan reconciliation...")
        print(f"Mode: {mode_str} | Grace Period: {args.grace_period}s | Quarantine: {args.quarantine} | Vault: {settings.PRIVATE_STORAGE_PATH}")

        with SessionLocal() as db_session:
            orphans_found = evidence_service.reconcile_orphan_evidence(
                db=db_session,
                dry_run=is_dry_run,
                grace_period_seconds=args.grace_period,
                quarantine=args.quarantine,
                quarantine_retention_seconds=args.quarantine_retention,
            )
            print(f"Reconciliation completed. Found {len(orphans_found)} orphan candidate files: {orphans_found}")
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)
