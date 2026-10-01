"""Reusable FastAPI dependencies for application services."""

from collections.abc import Iterator
from functools import lru_cache

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.database import create_database_engine, create_session_factory, get_db_session
from app.core.redis_client import create_redis_client
from app.services.redis_controls import RedisControls


@lru_cache(maxsize=1)
def _session_factory():
    """Create one process-wide session factory after database settings are loaded."""
    return create_session_factory(create_database_engine())


def get_database_session() -> Iterator[Session]:
    """Provide a request-scoped database session or a safe configuration error."""
    try:
        factory = _session_factory()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="The metadata database is not configured.") from exc
    yield from get_db_session(factory)


@lru_cache(maxsize=1)
def get_redis_controls() -> RedisControls:
    """Return one Redis-backed control adapter, or a local development fallback."""
    return RedisControls(create_redis_client())
