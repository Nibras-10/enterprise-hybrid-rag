"""Optional Redis client factory for cache and rate limiting."""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def create_redis_client() -> Any | None:
    """Return a configured Redis client, or None when Redis is not configured."""
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    from redis import Redis

    connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS", "10"))
    socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "10"))
    if connect_timeout <= 0 or socket_timeout <= 0:
        raise ValueError("Redis connection and socket timeouts must be greater than zero.")
    return Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=connect_timeout,
        socket_timeout=socket_timeout,
    )
