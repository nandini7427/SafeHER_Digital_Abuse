
"""FastAPI routes for secure evidence upload, management,
integrity verification, download and deletion.
"""

from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import AuditLog, Case, Evidence, User
from backend.app.schemas import (
    EvidenceIntegrityResponse,
    EvidenceResponse,
)
from backend.app.security.dependencies import get_current_user
from backend.app.services.evidence import save_evidence_file
from backend.app.services.evidence_service import evidence_service


router = APIRouter(
    prefix="/evidence",
    tags=["Evidence Management"],
)


# ---------------------------------------------------------
# Upload evidence to an owned incident case
# ---------------------------------------------------------

@router.post(
    "/cases/{case_id}",
    status_code=status.HTTP_201_CREATED,
    summary="Upload evidence to an owned case",
)
async def upload_evidence(
    case_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload an evidence file and record its integrity metadata."""

    original_filename = (
        (file.filename or "")
        .replace("\\", "/")
        .split("/")[-1]
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

    # Verify that the case belongs to the authenticated user.
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

    # MIME type is client-provided metadata.
    mime_type = file.content_type or "application/octet-stream"

    if len(mime_type) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type.",
        )

    storage_key = None

    try:
        storage_key, file_size, file_hash = await save_evidence_file(file)

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

        # Safely remove an uploaded file if the database operation failed.
        if storage_key:
            try:
                storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
                storage_path = (storage_dir / storage_key).resolve()

                # Only remove a direct child of the private storage directory.
                if storage_path.parent == storage_dir:
                    storage_path.unlink(missing_ok=True)

            except OSError:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence upload failed.",
        ) from exc

    finally:
        await file.close()


# ---------------------------------------------------------
# List evidence belonging to an owned case
# ---------------------------------------------------------

@router.get(
    "/cases/{case_id}",
    summary="List evidence associated with an owned case",
)
def list_case_evidence(
    case_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return sanitized evidence records for an owned case."""

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


# ---------------------------------------------------------
# Retrieve sanitized evidence metadata
# ---------------------------------------------------------

@router.get(
    "/{evidence_id}",
    response_model=EvidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get evidence metadata",
    description=(
        "Retrieves technical and cryptographic metadata for an "
        "owned evidence record without exposing internal storage paths."
    ),
)
def get_evidence_metadata(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceResponse:
    """Retrieve sanitized evidence metadata."""

    return evidence_service.get_evidence_metadata(
        evidence_id=evidence_id,
        current_user=current_user,
        db=db,
    )


# ---------------------------------------------------------
# Verify SHA-256 evidence integrity
# ---------------------------------------------------------

@router.get(
    "/{evidence_id}/verify",
    response_model=EvidenceIntegrityResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify evidence integrity",
    description=(
        "Recomputes the SHA-256 hash of the stored evidence file "
        "and compares it with the baseline hash recorded at upload."
    ),
)
def verify_evidence_integrity(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EvidenceIntegrityResponse:
    """Execute cryptographic integrity verification."""

    return evidence_service.verify_integrity(
        evidence_id=evidence_id,
        current_user=current_user,
        db=db,
    )


# ---------------------------------------------------------
# Download original evidence
# ---------------------------------------------------------

@router.get(
    "/{evidence_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Download original evidence",
    description=(
        "Streams the original stored evidence file to its "
        "authenticated and authorized owner."
    ),
)
def download_evidence(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download the original evidence file."""

    path, mime_type, original_filename = (
        evidence_service.retrieve_evidence_stream(
            evidence_id=evidence_id,
            current_user=current_user,
            db=db,
        )
    )

    return FileResponse(
        path=path,
        media_type=mime_type,
        filename=original_filename,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------
# Permanently delete owned evidence
# ---------------------------------------------------------

@router.delete(
    "/{evidence_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete owned evidence",
    description=(
        "Deletes an owned evidence record and its physical file "
        "through the controlled evidence-management service."
    ),
)
def delete_evidence(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete evidence record and its associated stored file."""

    evidence_service.delete_evidence(
        evidence_id=evidence_id,
        current_user=current_user,
        db=db,
    )

    return None
