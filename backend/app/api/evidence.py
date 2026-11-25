"""FastAPI route handlers for SafeHER Evidence Management, Integrity Audits, and Deletion."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import EvidenceIntegrityResponse, EvidenceResponse
from backend.app.security.dependencies import get_current_user
from backend.app.services.evidence_service import evidence_service

router = APIRouter(prefix="/evidence", tags=["Evidence Management"])


@router.get(
    "/{evidence_id}",
    response_model=EvidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get evidence metadata (strict horizontal ownership enforced)",
    description=(
        "Retrieves cryptographic and technical metadata for an owned evidence record. "
        "Does NOT expose internal filesystem paths or server-side storage keys."
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


@router.get(
    "/{evidence_id}/verify",
    response_model=EvidenceIntegrityResponse,
    status_code=status.HTTP_200_OK,
    summary="Cryptographically verify evidence integrity against disk content",
    description=(
        "Recomputes the SHA-256 hash of the stored file on disk and verifies it against "
        "the baseline cryptographic hash recorded at upload time. Detects disk tampering or bitrot."
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


@router.get(
    "/{evidence_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Download original pristine evidence stream (strictly authorized)",
    description="Streams the original, un-mutated evidence file directly to the verified owner.",
)
def download_evidence(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download pristine evidence file."""
    path, mime_type, original_filename = evidence_service.retrieve_evidence_stream(
        evidence_id=evidence_id,
        current_user=current_user,
        db=db,
    )
    return FileResponse(
        path=path,
        media_type=mime_type,
        filename=original_filename,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.delete(
    "/{evidence_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Permanently delete an owned evidence record and wipe file from disk",
    description=(
        "Controlled deletion pathway: purges physical file from private vault storage, "
        "removes database entity, and records immutable audit log."
    ),
)
def delete_evidence(
    evidence_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete evidence record and wipe from disk."""
    evidence_service.delete_evidence(
        evidence_id=evidence_id,
        current_user=current_user,
        db=db,
    )
    return None
