"""Application configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv

from src.indexing.embeddings import DEFAULT_DENSE_MODEL

DenseSearchMode = Literal["exact", "hnsw"]
_REPOSITORY_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


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


@dataclass(frozen=True, slots=True)
class GroqSettings:
    """Optional hosted-generation credentials loaded on demand."""

    api_key: str | None = None

    @classmethod
    def from_environment(cls) -> "GroqSettings":
        """Load Groq credentials from the process or repository-root ``.env``."""
        load_dotenv(_REPOSITORY_ENV_FILE, override=False)
        api_key = os.getenv("GROQ_API_KEY")
        return cls(api_key=api_key.strip() if api_key and api_key.strip() else None)
