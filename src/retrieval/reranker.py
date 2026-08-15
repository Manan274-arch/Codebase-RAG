"""Local cross-encoder reranking over a bounded candidate pool."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from langchain_core.documents import Document

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANK_CANDIDATE_DEPTH = 50


class CandidateResult(Protocol):
    """Minimum result interface expected by the reranker."""

    document: Document


class CandidateRetriever(Protocol):
    """First-stage interface expected by the reranker."""

    def retrieve(self, query: str, k: int = 10) -> Sequence[Any]: ...


class PairScorer(Protocol):
    """Injectable boundary for batched query/document pair scoring."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32]: ...


class RerankingError(ValueError):
    """Raised when a pair scorer returns unusable scores."""


@dataclass(frozen=True, slots=True)
class RerankedSearchResult:
    """A reranked original Document with preserved first-stage evidence."""

    document: Document
    score: float
    first_stage_score: float | None
    first_stage_rank: int
    candidate_provenance: str | None = None


class CrossEncoderReranker:
    """Rerank only candidates produced by the supplied first-stage retriever."""

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
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
        """Score first-stage candidates in one batch and return up to ``k`` results."""
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
                first_stage_score=_candidate_score(candidate),
                first_stage_rank=original_index + 1,
                candidate_provenance=_candidate_provenance(candidate),
            )
            for original_index, (candidate, score) in ranked[:k]
        ]


def _candidate_score(candidate: CandidateResult) -> float | None:
    score = getattr(candidate, "score", None)
    return float(score) if isinstance(score, (int, float)) else None


def _candidate_provenance(candidate: CandidateResult) -> str | None:
    provenance = getattr(candidate, "provenance", None)
    return provenance if isinstance(provenance, str) else None


class SentenceTransformerCrossEncoder:
    """Batched local Sentence Transformers CrossEncoder adapter."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32]:
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
