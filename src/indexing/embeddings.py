"""Shared local embedding boundary for indexing and dense queries."""

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

DEFAULT_DENSE_MODEL = "jinaai/jina-embeddings-v2-base-code"


class EmbeddingBackend(Protocol):
    """Normalized text-embedding interface shared by index and query paths."""

    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> npt.NDArray[np.float32]: ...


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


def normalized_matrix(
    embeddings: npt.ArrayLike, *, expected_rows: int
) -> npt.NDArray[np.float32]:
    """Validate and normalize an embedding matrix consistently across backends."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise ValueError(
            f"encoder must return shape ({expected_rows}, embedding_dimension)"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder returned a zero-length embedding")
    return cast(npt.NDArray[np.float32], matrix / norms)
