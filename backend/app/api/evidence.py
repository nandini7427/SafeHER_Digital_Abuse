
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
from backend.app.services.evidence import save_evidence_file


router = APIRouter(prefix="/evidence", tags=["Evidence"])

REQUIRED_EVIDENCE_CONSENT = "Evidence analysis"


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

    try:
        storage_key, file_size, file_hash = await save_evidence_file(file)

        duplicate = db.scalar(
            select(Evidence.id).where(
                Evidence.owner_id == current_user.id,
                Evidence.sha256 == file_hash,
            )
        )

        if duplicate is not None:
            storage_path = (
                Path(settings.PRIVATE_STORAGE_PATH).resolve()
                / storage_key
            )
            storage_path.unlink(missing_ok=True)
            storage_key = None

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Duplicate evidence detected. "
                    "This file has already been uploaded."
                ),
            )

        evidence = Evidence(
            case_id=case.id,
            owner_id=current_user.id,
            original_filename=original_filename,
            storage_key=storage_key,
            mime_type=mime_type,
            size_bytes=file_size,
            sha256=file_hash,
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
        db.refresh(evidence)

        return {
            "message": "Evidence uploaded successfully.",
            "id": evidence.id,
            "case_id": evidence.case_id,
            "original_filename": evidence.original_filename,
            "mime_type": evidence.mime_type,
            "size_bytes": evidence.size_bytes,
            "sha256": evidence.sha256,
            "uploaded_at": evidence.uploaded_at,
        }

    except HTTPException:
        db.rollback()

        if storage_key:
            storage_path = (
                Path(settings.PRIVATE_STORAGE_PATH).resolve()
                / storage_key
            )
            storage_path.unlink(missing_ok=True)

        raise

    except ValueError as exc:
        db.rollback()

        if storage_key:
            storage_path = (
                Path(settings.PRIVATE_STORAGE_PATH).resolve()
                / storage_key
            )
            storage_path.unlink(missing_ok=True)

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

        if storage_key:
            storage_path = (
                Path(settings.PRIVATE_STORAGE_PATH).resolve()
                / storage_key
            )
            storage_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence upload failed.",
        ) from exc

    finally:
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
