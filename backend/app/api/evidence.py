
import hashlib
import hmac
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import AuditLog, Case, Evidence, User
from backend.app.security.dependencies import get_current_user
from backend.app.services.consent import has_active_consent
from backend.app.services.evidence import (
    calculate_hash_distance,
    generate_perceptual_hash,
    save_evidence_file,
)


router = APIRouter(prefix="/evidence", tags=["Evidence"])

REQUIRED_EVIDENCE_CONSENT = "Evidence analysis"
PERCEPTUAL_HASH_DISTANCE_THRESHOLD = 8


# --------------------------------------------------
# 1. Upload evidence with consent and duplicate detection
# --------------------------------------------------

@router.post(
    "/cases/{case_id}",
    status_code=status.HTTP_201_CREATED,
)
async def upload_evidence(
    case_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    original_filename = (
        (file.filename or "").replace("\\", "/").split("/")[-1]
    )

    if (
        not original_filename
        or original_filename in {".", ".."}
        or len(original_filename) > 255
        or any(ord(char) < 32 for char in original_filename)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    case = db.scalar(
        select(Case).where(
            Case.id == case_id,
            Case.owner_id == current_user.id,
        )
    )

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )

    if not has_active_consent(
        db=db,
        user_id=current_user.id,
        purpose=REQUIRED_EVIDENCE_CONSENT,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Active consent for Evidence analysis is required "
                "before uploading evidence."
            ),
        )

    mime_type = file.content_type or "application/octet-stream"

    if len(mime_type) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type.",
        )

    storage_key = None
    upload_committed = False

    try:
        storage_key, file_size, file_hash = await save_evidence_file(file)

        storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
        storage_path = (storage_dir / storage_key).resolve()

        # Preserve exact SHA-256 duplicate detection.
        duplicate = db.scalar(
            select(Evidence.id).where(
                Evidence.owner_id == current_user.id,
                Evidence.sha256 == file_hash,
            )
        )

        if duplicate is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Duplicate evidence detected. "
                    "This file has already been uploaded."
                ),
            )

        # Generate pHash for supported image files.
        extension = Path(original_filename).suffix.lower()

        perceptual_hash = generate_perceptual_hash(
            storage_path,
            extension,
        )

        # Compare only with this user's existing image evidence.
        similar_evidence = []

        if perceptual_hash is not None:
            existing_evidence = db.scalars(
                select(Evidence).where(
                    Evidence.owner_id == current_user.id,
                    Evidence.perceptual_hash.is_not(None),
                )
            ).all()

            for existing in existing_evidence:
                distance = calculate_hash_distance(
                    perceptual_hash,
                    existing.perceptual_hash,
                )

                if distance <= PERCEPTUAL_HASH_DISTANCE_THRESHOLD:
                    similar_evidence.append(
                        {
                            "evidence_id": str(existing.id),
                            "original_filename": existing.original_filename,
                            "hamming_distance": distance,
                        }
                    )

        similarity_warning = len(similar_evidence) > 0

        evidence = Evidence(
            case_id=case.id,
            owner_id=current_user.id,
            original_filename=original_filename,
            storage_key=storage_key,
            mime_type=mime_type,
            size_bytes=file_size,
            sha256=file_hash,
            perceptual_hash=perceptual_hash,
        )

        db.add(evidence)
        db.flush()

        audit_entry = AuditLog(
            user_id=current_user.id,
            action="evidence_uploaded",
            entity_type="evidence",
            entity_id=str(evidence.id),
        )

        db.add(audit_entry)
        db.commit()

        # Mark successful upload so its stored file is preserved.
        upload_committed = True

        db.refresh(evidence)

        return {
            "message": "Evidence uploaded successfully.",
            "id": evidence.id,
            "case_id": evidence.case_id,
            "original_filename": evidence.original_filename,
            "mime_type": evidence.mime_type,
            "size_bytes": evidence.size_bytes,
            "sha256": evidence.sha256,
            "perceptual_hash": evidence.perceptual_hash,
            "similarity_warning": similarity_warning,
            "similar_evidence": similar_evidence,
            "uploaded_at": evidence.uploaded_at,
        }

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                if "upload limit" in str(exc).lower()
                else status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence upload failed.",
        ) from exc

    finally:
        # Delete the stored file only if the upload was not committed.
        if storage_key and not upload_committed:
            storage_path = (
                Path(settings.PRIVATE_STORAGE_PATH).resolve()
                / storage_key
            )
            storage_path.unlink(missing_ok=True)

        await file.close()


# --------------------------------------------------
# 2. List evidence belonging to a case
# --------------------------------------------------

@router.get("/cases/{case_id}")
def list_case_evidence(
    case_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = db.scalar(
        select(Case).where(
            Case.id == case_id,
            Case.owner_id == current_user.id,
        )
    )

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )

    evidence_records = db.scalars(
        select(Evidence)
        .where(
            Evidence.case_id == case_id,
            Evidence.owner_id == current_user.id,
        )
        .order_by(Evidence.uploaded_at.desc())
    ).all()

    return {
        "case_id": case_id,
        "total": len(evidence_records),
        "evidence": [
            {
                "id": item.id,
                "original_filename": item.original_filename,
                "mime_type": item.mime_type,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "perceptual_hash": item.perceptual_hash,
                "uploaded_at": item.uploaded_at,
            }
            for item in evidence_records
        ],
    }


# --------------------------------------------------
# 3. Download evidence with SHA-256 integrity verification
# --------------------------------------------------

@router.get("/{evidence_id}/download")
def download_evidence(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    evidence = db.scalar(
        select(Evidence).where(
            Evidence.id == evidence_id,
            Evidence.owner_id == current_user.id,
        )
    )

    if evidence is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found.",
        )

    storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
    storage_path = (storage_dir / evidence.storage_key).resolve()

    if storage_path.parent != storage_dir:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence file not found.",
        )

    if not storage_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence file not found.",
        )

    # Calculate SHA-256 from the stored evidence file.
    sha256_hash = hashlib.sha256()

    try:
        with storage_path.open("rb") as evidence_file:
            for chunk in iter(
                lambda: evidence_file.read(1024 * 1024),
                b"",
            ):
                sha256_hash.update(chunk)

    except OSError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to verify evidence integrity.",
        )

    calculated_hash = sha256_hash.hexdigest()

    # Block download if the file has been modified or has no stored hash.
    if (
        not evidence.sha256
        or not hmac.compare_digest(
            calculated_hash,
            evidence.sha256,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Evidence integrity verification failed. "
                "Download blocked."
            ),
        )

    download_filename = (
        (evidence.original_filename or "")
        .replace("\\", "/")
        .split("/")[-1]
    )

    if (
        not download_filename
        or any(
            ord(char) < 32 or ord(char) == 127
            for char in download_filename
        )
    ):
        download_filename = "evidence"

    return FileResponse(
        path=storage_path,
        media_type="application/octet-stream",
        filename=download_filename,
        content_disposition_type="attachment",
    )
