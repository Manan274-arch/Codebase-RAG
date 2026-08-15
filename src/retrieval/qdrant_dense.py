"""Query-only dense retrieval from an existing Qdrant code index."""

from dataclasses import dataclass

from langchain_core.documents import Document

from src.indexing.embeddings import EmbeddingBackend, normalized_matrix
from src.indexing.qdrant_index import QdrantCodeIndex, document_from_payload


@dataclass(frozen=True, slots=True)
class QdrantDenseSearchResult:
    """One reconstructed corpus document and its Qdrant cosine score."""

    document: Document
    score: float


class QdrantDenseRetriever:
    """Embed only the query and search a prebuilt persistent collection."""

    def __init__(
        self,
        index: QdrantCodeIndex,
        encoder: EmbeddingBackend,
        *,
        search_mode: str | None = None,
        hnsw_ef: int | None = None,
    ) -> None:
        self._index = index
        self._encoder = encoder
        self._search_mode = search_mode
        self._hnsw_ef = hnsw_ef

    def retrieve(self, query: str, k: int = 10) -> list[QdrantDenseSearchResult]:
        """Return dense results without rebuilding or re-embedding the corpus."""
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not query.strip():
            return []
        raw = self._encoder.encode([query], normalize_embeddings=True)
        vector = normalized_matrix(raw, expected_rows=1)[0]
        points = self._index.query(
            vector, k=k, mode=self._search_mode, hnsw_ef=self._hnsw_ef
        )
        return [
            QdrantDenseSearchResult(
                document=document_from_payload(point.payload), score=float(point.score)
            )
            for point in points
        ]
