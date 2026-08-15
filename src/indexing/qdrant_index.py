"""Persistent local Qdrant lifecycle for dense code embeddings."""

import hashlib
import json
import math
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import QdrantSettings
from src.indexing.embeddings import EmbeddingBackend, normalized_matrix
from src.retrieval.contracts import canonical_chunk_id

INDEX_SCHEMA_VERSION = 1
_POINT_NAMESPACE = uuid.UUID("00d6684c-c971-4f00-937a-9b43bda0958c")


class QdrantIndexError(ValueError):
    """Raised when a persistent collection is missing or incompatible."""


class StaleQdrantIndexError(QdrantIndexError):
    """Raised when an existing collection does not match the requested corpus."""


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Index lifecycle outcome with build phases kept separate from query timing."""

    reused: bool
    point_count: int
    embedding_seconds: float
    upsert_seconds: float
    fingerprint: str


class QdrantCodeIndex:
    """Create, validate, reuse, and query one persistent local Qdrant collection."""

    def __init__(
        self,
        settings: QdrantSettings,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        _validate_settings(settings)
        self.settings = settings
        self.client = client or QdrantClient(path=str(settings.path))

    @property
    def supports_real_hnsw(self) -> bool:
        """Local Python mode uses NumPy full scans and has no real HNSW graph."""
        return False

    def ensure(
        self,
        documents: Sequence[Document],
        encoder: EmbeddingBackend,
        *,
        rebuild: bool = False,
    ) -> IndexBuildResult:
        """Reuse a matching index or explicitly rebuild an absent/stale collection."""
        chunk_ids = [canonical_chunk_id(document) for document in documents]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise QdrantIndexError("corpus chunk identities must be unique")
        fingerprint = corpus_fingerprint(documents, self.settings)
        exists = self.client.collection_exists(self.settings.collection_name)
        if exists and not rebuild:
            self._validate_existing(documents, fingerprint)
            return IndexBuildResult(True, len(documents), 0.0, 0.0, fingerprint)
        if exists:
            self.client.delete_collection(self.settings.collection_name)
        self._create_collection(fingerprint)
        if not documents:
            return IndexBuildResult(False, 0, 0.0, 0.0, fingerprint)

        started = perf_counter()
        raw = encoder.encode(
            [document.page_content for document in documents],
            normalize_embeddings=True,
        )
        try:
            vectors = normalized_matrix(raw, expected_rows=len(documents))
        except ValueError as error:
            raise QdrantIndexError(str(error)) from error
        if vectors.shape[1] != self.settings.vector_size:
            raise QdrantIndexError(
                f"embedding dimension {vectors.shape[1]} does not match configured "
                f"dimension {self.settings.vector_size}"
            )
        embedding_seconds = perf_counter() - started
        points = [
            models.PointStruct(
                id=point_id(canonical_chunk_id(document)),
                vector=vector.tolist(),
                payload=document_payload(document),
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        started = perf_counter()
        self.client.upsert(
            collection_name=self.settings.collection_name,
            points=points,
            wait=True,
        )
        upsert_seconds = perf_counter() - started
        self._validate_existing(documents, fingerprint)
        return IndexBuildResult(
            False,
            len(points),
            embedding_seconds,
            upsert_seconds,
            fingerprint,
        )

    def query(
        self,
        vector: np.ndarray[Any, np.dtype[np.float32]],
        *,
        k: int,
        mode: str | None = None,
        hnsw_ef: int | None = None,
    ) -> list[models.ScoredPoint]:
        """Query an existing collection without embedding or rebuilding the corpus."""
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return []
        selected_mode = mode or self.settings.search_mode
        if selected_mode not in ("exact", "hnsw"):
            raise ValueError("search mode must be exact or hnsw")
        if vector.ndim != 1 or vector.shape[0] != self.settings.vector_size:
            raise QdrantIndexError("query vector has incompatible dimension")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Local mode performs exact.*search_params.*",
                category=UserWarning,
            )
            response = self.client.query_points(
                collection_name=self.settings.collection_name,
                query=vector.tolist(),
                search_params=models.SearchParams(
                    exact=selected_mode == "exact",
                    hnsw_ef=hnsw_ef or self.settings.hnsw_ef,
                ),
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
        return response.points

    def close(self) -> None:
        """Release the local storage lock."""
        self.client.close()

    def _create_collection(self, fingerprint: str) -> None:
        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.vector_size,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=self.settings.hnsw_m,
                ef_construct=self.settings.hnsw_ef_construct,
            ),
            metadata={
                "schema_version": INDEX_SCHEMA_VERSION,
                "corpus_fingerprint": fingerprint,
                "embedding_model": self.settings.embedding_model,
            },
        )

    def _validate_existing(
        self, documents: Sequence[Document], fingerprint: str
    ) -> None:
        info = self.client.get_collection(self.settings.collection_name)
        metadata = info.config.metadata or {}
        expected = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "corpus_fingerprint": fingerprint,
            "embedding_model": self.settings.embedding_model,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise StaleQdrantIndexError(
                "Qdrant collection does not match the corpus/embedding configuration; "
                "explicit rebuild required"
            )
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise StaleQdrantIndexError("Qdrant collection must use one unnamed vector")
        if (
            vectors.size != self.settings.vector_size
            or vectors.distance != models.Distance.COSINE
        ):
            raise StaleQdrantIndexError(
                "Qdrant vector dimension/distance is incompatible; explicit rebuild "
                "required"
            )
        count = self.client.count(
            collection_name=self.settings.collection_name, exact=True
        ).count
        if count != len(documents):
            raise StaleQdrantIndexError(
                f"Qdrant point count {count} does not match corpus size "
                f"{len(documents)}"
            )
        expected_ids = {
            canonical_chunk_id(document): point_id(canonical_chunk_id(document))
            for document in documents
        }
        actual_ids: dict[str, str] = {}
        offset: models.ExtendedPointId | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.settings.collection_name,
                limit=256,
                offset=offset,
                with_payload=["canonical_id"],
                with_vectors=False,
            )
            for record in records:
                canonical = (record.payload or {}).get("canonical_id")
                if not isinstance(canonical, str):
                    raise StaleQdrantIndexError(
                        "Qdrant point is missing canonical identity"
                    )
                actual_ids[canonical] = str(record.id)
            if offset is None:
                break
        if actual_ids != expected_ids:
            raise StaleQdrantIndexError(
                "Qdrant point identities do not match the corpus; explicit rebuild "
                "required"
            )


def corpus_fingerprint(documents: Sequence[Document], settings: QdrantSettings) -> str:
    """Hash ordered corpus content/metadata and embedding/index compatibility fields."""
    records = [
        {
            "id": canonical_chunk_id(document),
            "page_content": document.page_content,
            "metadata": _json_value(document.metadata),
            "document_id": document.id,
        }
        for document in documents
    ]
    material = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": settings.embedding_model,
        "vector_size": settings.vector_size,
        "distance": "cosine",
        "documents": records,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def point_id(chunk_id: str) -> str:
    """Derive a deterministic Qdrant-compatible UUID from canonical identity."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def document_payload(document: Document) -> dict[str, Any]:
    """Build an explicitly JSON-compatible payload for exact reconstruction."""
    return {
        "canonical_id": canonical_chunk_id(document),
        "source": document.metadata["source"],
        "chunk_index": document.metadata["chunk_index"],
        "page_content": document.page_content,
        "metadata": _json_value(document.metadata),
        "document_id": document.id,
    }


def document_from_payload(payload: Mapping[str, Any] | None) -> Document:
    """Reconstruct a LangChain Document and verify its canonical identity."""
    if payload is None:
        raise QdrantIndexError("Qdrant result is missing its payload")
    content = payload.get("page_content")
    metadata = payload.get("metadata")
    canonical = payload.get("canonical_id")
    document_id = payload.get("document_id")
    if not isinstance(content, str) or not isinstance(metadata, dict):
        raise QdrantIndexError("Qdrant payload cannot reconstruct a Document")
    if document_id is not None and not isinstance(document_id, str):
        raise QdrantIndexError("Qdrant payload has an invalid Document ID")
    document = Document(page_content=content, metadata=metadata, id=document_id)
    if canonical_chunk_id(document) != canonical:
        raise QdrantIndexError("Qdrant payload canonical identity is inconsistent")
    return document


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QdrantIndexError("metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise QdrantIndexError(f"metadata value is not JSON-compatible: {type(value)}")


def _validate_settings(settings: QdrantSettings) -> None:
    if not isinstance(settings.path, Path):
        raise ValueError("Qdrant path must be a Path")
    if not settings.collection_name:
        raise ValueError("Qdrant collection name must not be empty")
    for name in ("vector_size", "hnsw_m", "hnsw_ef_construct", "hnsw_ef"):
        value = getattr(settings, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
