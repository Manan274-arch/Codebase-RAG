"""Local raw-source code-aware dense retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from langchain_core.documents import Document

DEFAULT_DENSE_MODEL = "jinaai/jina-embeddings-v2-base-code"


class EmbeddingBackend(Protocol):
    """Injectable normalized text-embedding boundary."""

    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> npt.NDArray[np.float32]: ...


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
            self._corpus_embeddings = _normalized_matrix(
                raw_embeddings, expected_rows=len(self._documents)
            )
        else:
            self._corpus_embeddings = np.empty((0, 0), dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[DenseSearchResult]:
        """Return up to ``k`` results by cosine score, then corpus position."""
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not query.strip() or not self._documents:
            return []

        raw_query = self._encoder.encode([query], normalize_embeddings=True)
        query_embedding = _normalized_matrix(raw_query, expected_rows=1)
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


class SentenceTransformerBackend:
    """Local Sentence Transformers adapter for the selected code model."""

    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> npt.NDArray[np.float32]:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def _normalized_matrix(
    embeddings: npt.ArrayLike, *, expected_rows: int
) -> npt.NDArray[np.float32]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise DenseRetrievalError(
            f"encoder must return shape ({expected_rows}, embedding_dimension)"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise DenseRetrievalError("encoder returned a zero-length embedding")
    return cast(npt.NDArray[np.float32], matrix / norms)
