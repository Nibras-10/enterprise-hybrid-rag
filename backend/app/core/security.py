"""Single-tenant request authentication for local and managed deployments."""

import hmac
import os
from uuid import UUID

from fastapi import Header, HTTPException


def get_tenant_id(authorization: str | None = Header(default=None)) -> UUID:
    """Resolve the configured tenant; production requires a matching bearer token."""
    expected_token = os.getenv("API_BEARER_TOKEN", "").strip()
    environment = os.getenv("APP_ENV", "development").casefold()
    if expected_token:
        scheme, _, supplied_token = (authorization or "").partition(" ")
        if scheme.casefold() != "bearer" or not hmac.compare_digest(supplied_token, expected_token):
            raise HTTPException(status_code=401, detail="Authentication is required.")
    elif environment == "production":
        raise HTTPException(status_code=503, detail="Production authentication is not configured.")

    configured_tenant = os.getenv("DEFAULT_TENANT_ID", "00000000-0000-0000-0000-000000000001")
    try:
        return UUID(configured_tenant)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Tenant configuration is invalid.") from exc
