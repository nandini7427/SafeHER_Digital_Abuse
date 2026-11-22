
from fastapi import FastAPI

from backend.app.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered digital abuse intelligence and evidence response platform.",
    version="1.0.0"
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
