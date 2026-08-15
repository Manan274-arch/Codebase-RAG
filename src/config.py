"""Application configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from src.indexing.embeddings import DEFAULT_DENSE_MODEL

DenseSearchMode = Literal["exact", "hnsw"]


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    """Local persistent dense-index configuration."""

    path: Path = Path(".qdrant")
    collection_name: str = "code_chunks"
    embedding_model: str = DEFAULT_DENSE_MODEL
    vector_size: int = 768
    search_mode: DenseSearchMode = "exact"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef: int = 128

    @classmethod
    def from_environment(cls) -> "QdrantSettings":
        """Load the small set of settings needed by the local Python workflow."""
        mode = os.getenv("CODEBASE_RAG_DENSE_SEARCH_MODE", "exact")
        if mode not in ("exact", "hnsw"):
            raise ValueError("CODEBASE_RAG_DENSE_SEARCH_MODE must be exact or hnsw")
        return cls(
            path=Path(os.getenv("CODEBASE_RAG_QDRANT_PATH", ".qdrant")),
            collection_name=os.getenv("CODEBASE_RAG_QDRANT_COLLECTION", "code_chunks"),
            search_mode=cast(DenseSearchMode, mode),
        )
