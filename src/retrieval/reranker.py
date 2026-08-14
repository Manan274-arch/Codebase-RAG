"""Local cross-encoder reranking over RRF candidate results."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from langchain_core.documents import Document

from src.retrieval.hybrid import RRFSearchResult

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANK_CANDIDATE_DEPTH = 50


class RRFCandidateRetriever(Protocol):
    """First-stage interface expected by the reranker."""

    def retrieve(self, query: str, k: int = 10) -> Sequence[RRFSearchResult]: ...


class PairScorer(Protocol):
    """Injectable boundary for batched query/document pair scoring."""

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> npt.NDArray[np.float32]: ...


class RerankingError(ValueError):
    """Raised when a pair scorer returns unusable scores."""


@dataclass(frozen=True, slots=True)
class RerankedSearchResult:
    """A reranked original Document with preserved first-stage evidence."""

    document: Document
    score: float
    rrf_score: float
    rrf_rank: int


class CrossEncoderReranker:
    """Rerank only candidates produced by an RRF retriever."""

    def __init__(
        self,
        candidate_retriever: RRFCandidateRetriever,
        *,
        scorer: PairScorer | None = None,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        candidate_depth: int = DEFAULT_RERANK_CANDIDATE_DEPTH,
    ) -> None:
        _validate_positive_integer(candidate_depth, "candidate_depth")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        self._candidate_retriever = candidate_retriever
        self._scorer = scorer or SentenceTransformerCrossEncoder(model_name)
        self._candidate_depth = candidate_depth

    def retrieve(self, query: str, k: int = 10) -> list[RerankedSearchResult]:
        """Score RRF candidates in one batch and return up to ``k`` results."""
        _validate_non_negative_integer(k, "k")
        if k == 0 or not query.strip():
            return []
        candidates = list(
            self._candidate_retriever.retrieve(query, k=self._candidate_depth)
        )
        if not candidates:
            return []

        pairs = [(query, item.document.page_content) for item in candidates]
        scores = np.asarray(self._scorer.score(pairs), dtype=np.float32)
        if scores.ndim != 1 or scores.shape[0] != len(candidates):
            raise RerankingError(
                f"scorer must return shape ({len(candidates)},), got {scores.shape}"
            )
        if not np.all(np.isfinite(scores)):
            raise RerankingError("scorer returned a non-finite relevance score")

        ranked = sorted(
            enumerate(zip(candidates, scores, strict=True)),
            key=lambda item: (-float(item[1][1]), item[0]),
        )
        return [
            RerankedSearchResult(
                document=candidate.document,
                score=float(score),
                rrf_score=candidate.score,
                rrf_rank=original_index + 1,
            )
            for original_index, (candidate, score) in ranked[:k]
        ]


class SentenceTransformerCrossEncoder:
    """Batched local Sentence Transformers CrossEncoder adapter."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> npt.NDArray[np.float32]:
        scores = self._model.predict(
            list(pairs),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32)


def _validate_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
