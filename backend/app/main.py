
from fastapi import FastAPI

from backend.app.config import settings
from backend.app.api.auth import router as auth_router
from backend.app.api.cases import router as cases_router
from backend.app.api.admin import router as admin_router
from backend.app.api.threat import router as threat_router
from backend.app.api.image import router as image_router
from backend.app.api.evidence import router as evidence_router
from backend.app.api.review import router as review_router
from backend.app.services.image_service import upload_admission_coordinator
from backend.app.upload_middleware import UploadAdmissionMiddleware


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered digital abuse intelligence and evidence response platform.",
    version="1.0.0",
)


# Bound image upload requests before Starlette creates/spools multipart
# UploadFile objects.
# Allow up to 1 MiB of multipart framing and fields in addition to the
# configured file limit. ImageService enforces the exact per-file limit.
app.add_middleware(
    UploadAdmissionMiddleware,
    coordinator_provider=lambda: upload_admission_coordinator,
    max_request_body_bytes=(
        settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ) + (1024 * 1024),
)


# Authentication
app.include_router(
    auth_router,
    prefix=settings.API_V1_PREFIX,
)

# Incident cases
app.include_router(
    cases_router,
    prefix=settings.API_V1_PREFIX,
)

# Administrative endpoints
app.include_router(
    admin_router,
    prefix=settings.API_V1_PREFIX,
)

# AI threat assessment
app.include_router(
    threat_router,
    prefix=settings.API_V1_PREFIX,
)

# AI image assessment
app.include_router(
    image_router,
    prefix=settings.API_V1_PREFIX,
)

# Evidence upload, metadata, integrity verification,
# download, listing and deletion
app.include_router(
    evidence_router,
    prefix=settings.API_V1_PREFIX,
)

# Human review workflow
app.include_router(
    review_router,
    prefix=settings.API_V1_PREFIX,
)


@app.middleware("http")
async def add_privacy_and_security_headers(request, call_next):
    """Apply privacy and security headers to all HTTP responses."""

    response = await call_next(request)

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, private"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"

    return response


@app.get("/")
def home():
    return {
        "message": "Welcome to SafeHER API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
    }
