
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import settings


# Create database engine
connect_args = {}

if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)


# Database session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Base class for database models
class Base(DeclarativeBase):
    pass


# Dependency for API database access
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
