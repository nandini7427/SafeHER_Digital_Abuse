
from fastapi import FastAPI

from backend.app.config import settings
from backend.app.api.auth import router as auth_router
from backend.app.api.admin import router as admin_router
from backend.app.api.evidence import router as evidence_router
from backend.app.api.cases import router as cases_router


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered digital abuse intelligence and evidence response platform.",
    version="1.0.0"
)


# Authentication routes
app.include_router(
    auth_router,
    prefix=settings.API_V1_PREFIX
)

# Admin routes
app.include_router(
    admin_router,
    prefix=settings.API_V1_PREFIX
)

# Evidence management routes
app.include_router(
    evidence_router,
    prefix=settings.API_V1_PREFIX
)

# Case management routes
app.include_router(
    cases_router,
    prefix=settings.API_V1_PREFIX
)


@app.get("/")
def home():
    return {
        "message": "Welcome to SafeHER API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME
    }
