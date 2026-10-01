"""Environment-backed settings needed by the upload workflow."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class UploadSettings:
    """Validated local settings for incoming document files."""

    storage_path: Path
    max_upload_size_bytes: int

    @classmethod
    def from_environment(cls) -> "UploadSettings":
        """Build settings from environment variables with safe local defaults."""
        max_upload_size = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", "20971520"))
        if max_upload_size <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be greater than zero.")

        storage_path = Path(os.getenv("DOCUMENT_STORAGE_PATH", "./data/uploads"))
        return cls(storage_path=storage_path.resolve(), max_upload_size_bytes=max_upload_size)
