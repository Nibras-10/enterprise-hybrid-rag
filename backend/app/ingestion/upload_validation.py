"""Filename and content validation for supported document uploads."""

from __future__ import annotations

import codecs
from pathlib import Path
import re
import zipfile


class UploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or has unsupported content."""


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


def sanitize_filename(filename: str | None) -> str:
    """Normalize an untrusted filename and reject path/control characters."""
    if not filename:
        raise UploadValidationError("A filename is required.")

    normalized = filename.replace("\\", "/").strip()
    basename = normalized.rsplit("/", maxsplit=1)[-1]
    basename = _CONTROL_CHARACTERS.sub("", basename).strip()

    if not basename or basename in {".", ".."}:
        raise UploadValidationError("The filename is invalid.")
    if len(basename) > 255:
        raise UploadValidationError("The filename must be 255 characters or fewer.")

    suffix = Path(basename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("Only PDF, DOCX, and TXT files are supported.")
    return basename


def validate_file_content(path: Path, extension: str) -> None:
    """Verify file signatures and basic structural content without trusting MIME headers."""
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("Only PDF, DOCX, and TXT files are supported.")

    try:
        if extension == ".pdf":
            _validate_pdf(path)
        elif extension == ".docx":
            _validate_docx(path)
        else:
            _validate_text(path)
    except OSError as exc:
        raise UploadValidationError("The uploaded file could not be read.") from exc


def _validate_pdf(path: Path) -> None:
    with path.open("rb") as file:
        header = file.read(1024)
    if b"%PDF-" not in header:
        raise UploadValidationError("The file extension does not match valid PDF content.")


def _validate_docx(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise UploadValidationError("The file extension does not match valid DOCX content.")

    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise UploadValidationError("The DOCX package is missing required document content.")
            if sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise UploadValidationError("The DOCX package expands beyond the supported size limit.")
            if any(entry.flag_bits & 0x1 for entry in entries):
                raise UploadValidationError("Encrypted DOCX packages are not supported.")
            required_entries = {entry.filename: entry for entry in entries}
            if any(required_entries[name].file_size == 0 for name in ("[Content_Types].xml", "word/document.xml")):
                raise UploadValidationError("The DOCX package contains empty required document content.")
    except (OSError, zipfile.BadZipFile) as exc:
        raise UploadValidationError("The DOCX package is damaged.") from exc


def _validate_text(path: Path) -> None:
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="strict")
    try:
        with path.open("rb") as file:
            while block := file.read(64 * 1024):
                if b"\x00" in block:
                    raise UploadValidationError("The file extension does not match plain text content.")
                decoder.decode(block)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise UploadValidationError("Text files must use UTF-8 encoding.") from exc
