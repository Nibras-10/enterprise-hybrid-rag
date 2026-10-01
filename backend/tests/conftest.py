"""Shared in-memory API fixtures."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Keep module-level middleware isolated from service credentials in a developer's .env.
os.environ["REDIS_URL"] = ""

from app.api.dependencies import get_database_session, get_redis_controls
from app.main import app
from app.models import Base


@pytest.fixture
def api_client(monkeypatch):
    """Provide a clean SQLite-backed API and local request controls per test."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("API_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    get_redis_controls.cache_clear()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    def override_database_session():
        with factory() as session:
            yield session

    app.dependency_overrides[get_database_session] = override_database_session
    with TestClient(app) as client:
        yield client, factory
    app.dependency_overrides.clear()
    get_redis_controls.cache_clear()
    Base.metadata.drop_all(engine)
    engine.dispose()
