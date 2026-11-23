from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User, Case, Evidence
from backend.app.security.permissions import require_roles


router = APIRouter(prefix="/admin", tags=["Administration"])


@router.get("/summary")
def get_admin_summary(
    current_admin: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    total_users = db.scalar(
        select(func.count()).select_from(User)
    )

    total_cases = db.scalar(
        select(func.count()).select_from(Case)
    )

    total_evidence = db.scalar(
        select(func.count()).select_from(Evidence)
    )

    return {
        "total_users": total_users,
        "total_cases": total_cases,
        "total_evidence": total_evidence,
    }
