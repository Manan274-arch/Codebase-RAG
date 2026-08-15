"""Rank-only Reciprocal Rank Fusion retained as an evaluated baseline."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from src.retrieval.contracts import RankedDocument, canonical_chunk_id

DEFAULT_RRF_CONSTANT = 60
DEFAULT_CANDIDATE_DEPTH = 50


class BranchRetriever(Protocol):
    """Ranked retrieval interface consumed by the hybrid retriever."""

    def retrieve(self, query: str, k: int = 10) -> Sequence[RankedDocument]: ...


@dataclass(frozen=True, slots=True)
class RRFSearchResult:
    """One original document and its rank-derived fused score."""

    document: Document
    score: float


@dataclass(slots=True)
class _AccumulatedResult:
    document: Document
    score: float
    best_rank: int
    first_seen: int


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RankedDocument]],
    *,
    top_k: int = 10,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[RRFSearchResult]:
    """Fuse ranked lists using ``sum(1 / (rrf_constant + 1-based rank))``.

    A canonical chunk contributes at most once per branch. Exact score ties are
    resolved by best individual rank, then first appearance by branch/list order.
    """
    _validate_non_negative_integer(top_k, "top_k")
    _validate_positive_integer(rrf_constant, "rrf_constant")
    if top_k == 0:
        return []

    accumulated: dict[str, _AccumulatedResult] = {}
    next_seen = 0
    for ranked in ranked_lists:
        seen_in_branch: set[str] = set()
        for rank, result in enumerate(ranked, start=1):
            chunk_id = canonical_chunk_id(result.document)
            if chunk_id in seen_in_branch:
                continue
            seen_in_branch.add(chunk_id)
            contribution = 1.0 / (rrf_constant + rank)
            existing = accumulated.get(chunk_id)
            if existing is None:
                accumulated[chunk_id] = _AccumulatedResult(
                    document=result.document,
                    score=contribution,
                    best_rank=rank,
                    first_seen=next_seen,
                )
                next_seen += 1
            else:
                existing.score += contribution
                existing.best_rank = min(existing.best_rank, rank)

    ordered = sorted(
        accumulated.values(),
        key=lambda item: (-item.score, item.best_rank, item.first_seen),
    )
    return [
        RRFSearchResult(document=item.document, score=item.score)
        for item in ordered[:top_k]
    ]


class HybridRetriever:
    """Retrieve independent branch candidates and fuse their rank positions."""

    def __init__(
        self,
        bm25_retriever: BranchRetriever,
        dense_retriever: BranchRetriever,
        *,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ) -> None:
        _validate_positive_integer(candidate_depth, "candidate_depth")
        _validate_positive_integer(rrf_constant, "rrf_constant")
        self._bm25_retriever = bm25_retriever
        self._dense_retriever = dense_retriever
        self._candidate_depth = candidate_depth
        self._rrf_constant = rrf_constant

    def retrieve(self, query: str, k: int = 10) -> list[RRFSearchResult]:
        """Return ``k`` fused results from separately sized branch candidate lists."""
        _validate_non_negative_integer(k, "k")
        if k == 0 or not query.strip():
            return []
        bm25_results = self._bm25_retriever.retrieve(query, k=self._candidate_depth)
        dense_results = self._dense_retriever.retrieve(query, k=self._candidate_depth)
        return reciprocal_rank_fusion(
            (bm25_results, dense_results),
            top_k=k,
            rrf_constant=self._rrf_constant,
        )


def _validate_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
