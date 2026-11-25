"""FastAPI route handlers for SafeHER AI Image Intelligence and Deepfake Detection."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import IntegratedForensicAssessment
from backend.app.security.dependencies import get_current_user
from backend.app.services.image_service import backend_image_service

router = APIRouter(prefix="/ai", tags=["AI Image Intelligence"])


@router.post(
    "/image-assessment",
    response_model=IntegratedForensicAssessment,
    status_code=status.HTTP_200_OK,
    summary="Assess uploaded image for deepfake manipulation and visual anomalies",
    description=(
        "Analyzes raw uploaded image evidence for synthetic manipulation, face swapping, "
        "and compression inconsistencies using Error Level Analysis (ELA) and visual forensics. "
        "Supports optional case association, strict horizontal authorization, and evidence preservation."
    ),
)
async def assess_image(
    file: UploadFile = File(..., description="Multipart image file upload"),
    case_id: Optional[UUID] = Form(None, description="Optional existing case ID to link this assessment with"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IntegratedForensicAssessment:
    """Execute authenticated visual manipulation assessment on an uploaded image."""
    return await backend_image_service.assess_image_upload(
        file=file,
        current_user=current_user,
        case_id=case_id,
        db=db,
    )


@router.post(
    "/image/assess",
    response_model=IntegratedForensicAssessment,
    status_code=status.HTTP_200_OK,
    summary="Assess uploaded image for deepfake manipulation (alias route)",
    include_in_schema=True,
)
async def assess_image_alias(
    file: UploadFile = File(..., description="Multipart image file upload"),
    case_id: Optional[UUID] = Form(None, description="Optional existing case ID to link this assessment with"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IntegratedForensicAssessment:
    """Alias endpoint for image manipulation assessment."""
    return await backend_image_service.assess_image_upload(
        file=file,
        current_user=current_user,
        case_id=case_id,
        db=db,
    )
