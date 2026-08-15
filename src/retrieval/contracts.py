"""Shared production retrieval contracts and canonical chunk identity."""

from typing import Protocol

from langchain_core.documents import Document


class RankedDocument(Protocol):
    """A ranked result containing an original or reconstructed Document."""

    @property
    def document(self) -> Document: ...


def canonical_chunk_id(document: Document) -> str:
    """Encode the established ``source + chunk_index`` chunk identity."""
    source = document.metadata.get("source")
    chunk_index = document.metadata.get("chunk_index")
    if not isinstance(source, str) or not source:
        raise ValueError("chunk identity requires non-empty string 'source' metadata")
    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or chunk_index < 0
    ):
        raise ValueError(
            f"chunk {source!r} requires non-negative integer 'chunk_index' metadata"
        )
    return f"{source}::{chunk_index}"
