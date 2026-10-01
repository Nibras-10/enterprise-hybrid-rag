"""Convert parser output to LlamaIndex nodes without dropping source structure."""

import json
from uuid import UUID

from app.ingestion.elements import DocumentElement


class LlamaIndexDocumentBridge:
    """Keep LlamaIndex node metadata aligned with the normalized parser model."""

    @staticmethod
    def to_nodes(elements: tuple[DocumentElement, ...]) -> list[object]:
        """Represent source elements as LlamaIndex TextNodes for framework workflows."""
        from llama_index.core.schema import TextNode

        nodes = []
        for element in elements:
            metadata = {
                "document_id": str(element.document_id),
                "page_number": element.page_number,
                "element_type": element.element_type,
                "section_title": element.section_title,
                "source_metadata_json": json.dumps(element.metadata, ensure_ascii=False),
            }
            nodes.append(TextNode(
                text=element.text,
                metadata={key: value for key, value in metadata.items() if value is not None},
            ))
        return nodes

    @staticmethod
    def to_elements(nodes: list[object]) -> tuple[DocumentElement, ...]:
        """Restore normalized elements from framework nodes before semantic chunking."""
        elements: list[DocumentElement] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {})
            try:
                source_metadata = json.loads(metadata.get("source_metadata_json", "{}"))
                elements.append(DocumentElement(
                    document_id=UUID(metadata["document_id"]),
                    page_number=metadata.get("page_number"),
                    element_type=str(metadata.get("element_type", "UncategorizedText")),
                    text=str(node.text),
                    section_title=metadata.get("section_title"),
                    metadata=source_metadata,
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("LlamaIndex node is missing valid source metadata.") from exc
        return tuple(elements)
