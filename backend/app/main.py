"""FastAPI application entry point and cross-cutting request controls."""

import logging
import os
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.documents import router as documents_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.queries import router as queries_router
from app.api.dependencies import get_redis_controls
from app.services.redis_controls import RateLimitExceeded, RedisControls
from app.core.settings import UploadSettings

logger = logging.getLogger(__name__)

_production = os.getenv("APP_ENV", "development").casefold() == "production"
if _production:
    _required_security_settings = {
        "API_BEARER_TOKEN": os.getenv("API_BEARER_TOKEN", "").strip(),
        "CORS_ALLOWED_ORIGINS": os.getenv("CORS_ALLOWED_ORIGINS", "").strip(),
        "REDIS_URL": os.getenv("REDIS_URL", "").strip(),
    }
    _missing_security_settings = [name for name, value in _required_security_settings.items() if not value]
    if _missing_security_settings:
        raise RuntimeError(
            "Production security configuration is missing: "
            + ", ".join(_missing_security_settings)
        )
    if "*" in _required_security_settings["CORS_ALLOWED_ORIGINS"].split(","):
        raise RuntimeError("CORS_ALLOWED_ORIGINS must use explicit origins in production.")

app = FastAPI(title="Enterprise Hybrid RAG API", version="0.1.0")
app.include_router(documents_router)
app.include_router(queries_router)
app.include_router(evaluations_router)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

try:
    redis_controls = get_redis_controls()
except (ImportError, ValueError):
    if _production:
        raise RuntimeError("Redis is required for production request controls.")
    logger.exception("Redis configuration is invalid; using process-local controls")
    redis_controls = RedisControls(None)

if _production and redis_controls.client is None:
    raise RuntimeError("Redis is required for production request controls.")


@app.middleware("http")
async def request_controls(request: Request, call_next):
    """Assign request IDs and protect expensive API operations with rate limits."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    limited = (
        request.url.path == "/api/query" and request.method == "POST"
    ) or (
        request.url.path == "/api/documents" and request.method == "POST"
    ) or request.url.path == "/api/evaluations/run"
    if request.url.path == "/api/documents" and request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": "Content-Length is invalid."})
                response.headers["X-Request-ID"] = request_id
                return response
            maximum_request_size = UploadSettings.from_environment().max_upload_size_bytes + 1024 * 1024
            if declared_size > maximum_request_size:
                response = JSONResponse(status_code=413, content={"detail": "The uploaded request exceeds the configured size limit."})
                response.headers["X-Request-ID"] = request_id
                return response
    if limited:
        address = request.client.host if request.client else "unknown"
        try:
            redis_controls.enforce_rate_limit(address)
        except RateLimitExceeded:
            response = JSONResponse(status_code=429, content={"detail": "Request rate limit exceeded."})
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            logger.exception("Rate limiting service is unavailable")
            response = JSONResponse(status_code=503, content={"detail": "Request controls are unavailable."})
            response.headers["X-Request-ID"] = request_id
            return response
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {"status": "ok"}
