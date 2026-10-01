"""Privacy-conscious Langfuse spans for RAG operations."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class LangfuseSettings:
    enabled: bool
    public_key: str
    secret_key: str
    host: str | None = None

    @classmethod
    def from_environment(cls) -> "LangfuseSettings":
        enabled = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
        return cls(
            enabled=enabled,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST") or None,
        )


class LangfuseTracer:
    """Create traces only when explicitly enabled and credentials are configured."""

    def __init__(self, settings: LangfuseSettings | None = None, client: Any | None = None) -> None:
        self.settings = settings or LangfuseSettings.from_environment()
        self._client = client
        if self.settings.enabled and not (self.settings.public_key and self.settings.secret_key):
            raise ValueError("Langfuse requires both public and secret keys when enabled.")

    @contextmanager
    def span(self, name: str, *, metadata: dict[str, Any] | None = None) -> Iterator[Any | None]:
        """Yield a span or None; inputs/outputs are not captured by default."""
        if not self.settings.enabled:
            yield None
            return
        if self._client is None:
            from langfuse import get_client

            # The SDK reads credentials from environment variables.
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.settings.public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", self.settings.secret_key)
            if self.settings.host:
                os.environ.setdefault("LANGFUSE_HOST", self.settings.host)
            self._client = get_client()
        with self._client.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata or {},
        ) as observation:
            yield observation
