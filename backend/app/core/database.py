"""SQLAlchemy engine and request-scoped session helpers."""

import os
from collections.abc import Iterator

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a PostgreSQL engine from an explicit URL or environment setting."""
    url = database_url or os.getenv("SUPABASE_DATABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_DATABASE_URL must be configured before using the database.")
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError("SUPABASE_DATABASE_URL must use PostgreSQL.")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a reusable factory for short-lived database sessions."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Yield a session and guarantee rollback and cleanup on request failures."""
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
