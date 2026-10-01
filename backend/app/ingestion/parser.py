"""Structure-aware parsing adapter backed by Unstructured."""

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from app.ingestion.elements import DocumentElement, ParsedDocument


class DocumentParsingError(RuntimeError):
    """Raised when an uploaded file cannot be parsed into document elements."""


class UnstructuredDocumentParser:
    """Parse supported business documents without flattening their structure."""

    _SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}
    _SECTION_CATEGORIES = {"Title", "Heading", "Heading 1", "Heading 2", "Heading 3"}

    def parse(self, file_path: Path, document_id: UUID) -> ParsedDocument:
        """Extract paragraphs, titles, headings, tables, lists, and page metadata."""
        suffix = file_path.suffix.lower()
        if suffix not in self._SUPPORTED_SUFFIXES:
            raise DocumentParsingError("Unsupported document type for parsing.")
        if not file_path.is_file():
            raise DocumentParsingError("The document source file is unavailable.")

        try:
            raw_elements = self._partition(file_path, suffix)
            elements = self._normalize(raw_elements, document_id, file_path.name)
        except DocumentParsingError:
            raise
        except Exception as exc:
            raise DocumentParsingError("The document could not be parsed.") from exc

        if not elements:
            raise DocumentParsingError("The document contains no readable text or structure.")
        page_count = max((element.page_number or 0 for element in elements), default=0)
        return ParsedDocument(elements=tuple(elements), page_count=page_count)

    @staticmethod
    def _partition(file_path: Path, suffix: str) -> Iterable[Any]:
        """Select the format-specific Unstructured partitioner lazily."""
        if suffix == ".pdf":
            from unstructured.partition.pdf import partition_pdf

            return partition_pdf(
                filename=str(file_path),
                strategy="hi_res",
                infer_table_structure=True,
            )
        if suffix == ".docx":
            from unstructured.partition.docx import partition_docx

            return partition_docx(filename=str(file_path))

        from unstructured.partition.text import partition_text

        return partition_text(filename=str(file_path), encoding="utf-8")

    @classmethod
    def _normalize(
        cls,
        raw_elements: Iterable[Any],
        document_id: UUID,
        source_filename: str,
    ) -> list[DocumentElement]:
        """Map parser elements while carrying forward the nearest section heading."""
        normalized: list[DocumentElement] = []
        active_section: str | None = None
        table_count = 0

        for raw_element in raw_elements:
            category = str(getattr(raw_element, "category", "UncategorizedText"))
            metadata = cls._metadata_dict(raw_element)
            text = str(getattr(raw_element, "text", "") or "").strip()

            if category in cls._SECTION_CATEGORIES and text:
                active_section = text

            if category == "Table":
                table_count += 1
                table_html = metadata.get("text_as_html")
                if table_html:
                    text = str(table_html).strip()
                metadata["table_id"] = f"table-{table_count:04d}"

            if not text:
                continue

            page_number = metadata.get("page_number")
            if not isinstance(page_number, int):
                page_number = None

            metadata["source_filename"] = source_filename
            normalized.append(
                DocumentElement(
                    document_id=document_id,
                    page_number=page_number,
                    element_type=category,
                    text=text,
                    section_title=active_section,
                    metadata=metadata,
                )
            )

        return normalized

    @staticmethod
    def _metadata_dict(raw_element: Any) -> dict[str, Any]:
        """Return serializable parser metadata while preserving supported fields."""
        parser_metadata = getattr(raw_element, "metadata", None)
        to_dict: Callable[[], dict[str, Any]] | None = getattr(parser_metadata, "to_dict", None)
        if not callable(to_dict):
            return {}

        values = to_dict()
        serializable = json.loads(json.dumps(values, default=str, allow_nan=False))
        return {key: value for key, value in serializable.items() if value is not None}
