"""Exact in-memory dense reference retained for migration evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document

from src.indexing.embeddings import (
    EmbeddingBackend,
    SentenceTransformerBackend,
    normalized_matrix,
)


class DenseRetrievalError(ValueError):
    """Raised when an embedding backend returns unusable vectors."""


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    """One original corpus document and its cosine-similarity score."""

    document: Document
    score: float


class DenseRetriever:
    """Embed only raw ``page_content`` and rank by cosine similarity."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        encoder: EmbeddingBackend | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._encoder = encoder or SentenceTransformerBackend()
        if self._documents:
            raw_embeddings = self._encoder.encode(
                [document.page_content for document in self._documents],
                normalize_embeddings=True,
            )
            try:
                self._corpus_embeddings = normalized_matrix(
                    raw_embeddings, expected_rows=len(self._documents)
                )
            except ValueError as error:
                raise DenseRetrievalError(str(error)) from error
        else:
            self._corpus_embeddings = np.empty((0, 0), dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[DenseSearchResult]:
        """Return up to ``k`` results by cosine score, then corpus position."""
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not query.strip() or not self._documents:
            return []

        raw_query = self._encoder.encode([query], normalize_embeddings=True)
        try:
            query_embedding = normalized_matrix(raw_query, expected_rows=1)
        except ValueError as error:
            raise DenseRetrievalError(str(error)) from error
        if query_embedding.shape[1] != self._corpus_embeddings.shape[1]:
            raise DenseRetrievalError(
                "query and corpus embedding dimensions must match"
            )
        scores = self._corpus_embeddings @ query_embedding[0]
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [
            DenseSearchResult(
                document=self._documents[index],
                score=float(score),
            )
            for index, score in ranked[: min(k, len(self._documents))]
        ]
