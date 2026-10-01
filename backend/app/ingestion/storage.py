"""Bounded local staging and atomic storage for uploaded source files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from app.ingestion.upload_validation import UploadValidationError

READ_BLOCK_SIZE = 1024 * 1024


class UploadTooLargeError(UploadValidationError):
    """Raised when an upload exceeds the configured size limit."""


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """Temporary upload location and content fingerprint."""

    path: Path
    sha256: str
    size_bytes: int


class LocalDocumentStorage:
    """Store documents under generated IDs rather than user supplied paths."""

    def __init__(self, root: Path, max_size_bytes: int) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be greater than zero.")
        self.root = root.resolve()
        self.staging_root = self.root / ".staging"
        self.max_size_bytes = max_size_bytes

    def stage(self, source: BinaryIO) -> StagedUpload:
        """Copy a stream to a private staging file while enforcing size and hashing."""
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staged_path = self.staging_root / f"{os.urandom(16).hex()}.upload"
        digest = hashlib.sha256()
        total_bytes = 0

        try:
            with staged_path.open("xb") as destination:
                while block := source.read(READ_BLOCK_SIZE):
                    total_bytes += len(block)
                    if total_bytes > self.max_size_bytes:
                        raise UploadTooLargeError("The uploaded file exceeds the configured size limit.")
                    digest.update(block)
                    destination.write(block)
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise

        if total_bytes == 0:
            staged_path.unlink(missing_ok=True)
            raise UploadValidationError("The uploaded file is empty.")

        return StagedUpload(path=staged_path, sha256=digest.hexdigest(), size_bytes=total_bytes)

    def commit(self, staged: StagedUpload, document_id: UUID, extension: str) -> str:
        """Atomically move staged bytes into a document-ID keyed location."""
        relative_path = Path("documents") / str(document_id) / f"source{extension}"
        destination = (self.root / relative_path).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("Resolved document path escaped the configured storage root.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staged.path, destination)
        except Exception:
            self.discard(staged)
            raise
        return relative_path.as_posix()

    def discard(self, staged: StagedUpload) -> None:
        """Remove a staged file after rejection or a failed database transaction."""
        staged.path.unlink(missing_ok=True)

    def remove(self, storage_reference: str) -> None:
        """Remove a stored source file only when its resolved path stays under root."""
        target = (self.root / storage_reference).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Resolved document path escaped the configured storage root.")
        target.unlink(missing_ok=True)
        parent = target.parent
        if parent != self.root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                pass

    def resolve(self, storage_reference: str) -> Path:
        """Resolve a stored reference while preventing path traversal."""
        target = (self.root / storage_reference).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Resolved document path escaped the configured storage root.")
        return target
