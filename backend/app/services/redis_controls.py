"""Tenant-safe Redis caching and distributed fixed-window rate limiting."""

from hashlib import sha256
import json
import os
import threading
import time
from typing import Any
from uuid import UUID


class RateLimitExceeded(RuntimeError):
    """Raised when a caller exceeds the configured request budget."""


class RedisControls:
    """Small adapter that keeps Redis optional while scoping cache entries by tenant."""

    def __init__(
        self,
        client: Any | None,
        *,
        cache_ttl_seconds: int | None = None,
        rate_limit_requests: int | None = None,
        rate_limit_window_seconds: int | None = None,
        namespace: str = "enterprise-hybrid-rag",
    ) -> None:
        self.client = client
        self.cache_ttl_seconds = cache_ttl_seconds or int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self.rate_limit_requests = rate_limit_requests or int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
        self.rate_limit_window_seconds = rate_limit_window_seconds or int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        if min(self.cache_ttl_seconds, self.rate_limit_requests, self.rate_limit_window_seconds) <= 0:
            raise ValueError("Redis TTL and rate-limit settings must be greater than zero.")
        self.namespace = namespace
        self._local_cache: dict[str, tuple[float, str]] = {}
        self._local_counts: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def cache_key(
        self,
        *,
        tenant_id: UUID,
        question: str,
        document_ids: list[UUID] | None,
        config_version: str,
    ) -> str:
        scope = sorted(str(item) for item in (document_ids or []))
        canonical = json.dumps(
            {"question": question.strip(), "document_ids": scope, "config_version": config_version},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        return f"{self.namespace}:query:{tenant_id}:{digest}"

    def get_json(self, key: str) -> Any | None:
        if self.client is not None:
            value = self.client.get(key)
            return json.loads(value) if value else None
        with self._lock:
            cached = self._local_cache.get(key)
            if cached is None:
                return None
            if cached[0] <= time.monotonic():
                self._local_cache.pop(key, None)
                return None
            return json.loads(cached[1])

    def set_json(self, key: str, value: Any) -> None:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if self.client is not None:
            self.client.set(key, serialized, ex=self.cache_ttl_seconds)
            return
        with self._lock:
            self._local_cache[key] = (time.monotonic() + self.cache_ttl_seconds, serialized)

    def invalidate_tenant_cache(self, tenant_id: UUID) -> None:
        """Invalidate cached answers for a tenant after its searchable corpus changes."""
        prefix = f"{self.namespace}:query:{tenant_id}:"
        if self.client is not None:
            keys = list(self.client.scan_iter(match=f"{prefix}*", count=200))
            if keys:
                self.client.delete(*keys)
            return
        with self._lock:
            for key in [key for key in self._local_cache if key.startswith(prefix)]:
                self._local_cache.pop(key, None)

    def enforce_rate_limit(self, identity: str, *, now: int | None = None) -> None:
        """Apply a per-identity fixed window; Redis counters are shared across workers."""
        if not identity or len(identity) > 256:
            raise ValueError("Rate-limit identity is invalid.")
        window = (now if now is not None else int(time.time())) // self.rate_limit_window_seconds
        key = f"{self.namespace}:rate:{sha256(identity.encode()).hexdigest()}:{window}"
        if self.client is not None:
            count = int(self.client.incr(key))
            if count == 1:
                self.client.expire(key, self.rate_limit_window_seconds + 1)
        else:
            expires_at = (window + 1) * self.rate_limit_window_seconds
            with self._lock:
                count, expiry = self._local_counts.get(key, (0, expires_at))
                if expiry <= (now if now is not None else int(time.time())):
                    count = 0
                count += 1
                self._local_counts[key] = (count, expires_at)
        if count > self.rate_limit_requests:
            raise RateLimitExceeded("Request rate limit exceeded.")
