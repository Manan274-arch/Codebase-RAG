"""Compose ingestion stages into the single enriched retrieval corpus."""

from pathlib import Path

from langchain_core.documents import Document

from src.chunking.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
)
from src.enrichment.metadata import enrich_chunks
from src.enrichment.relationships import link_http_calls_to_routes
from src.enrichment.representation import enrich_retrieval_corpus
from src.enrichment.structure import extract_structure
from src.ingestion.loader import load_source_documents


def build_enriched_corpus(
    repository: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Build the relationship-linked, structurally rendered shared corpus."""
    documents = load_source_documents(repository)
    enriched: list[Document] = []
    for document in documents:
        chunks = chunk_documents(
            [document], chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        enriched.extend(enrich_chunks(chunks, extract_structure(document)))
    linked = link_http_calls_to_routes(enriched)
    return enrich_retrieval_corpus(linked)
