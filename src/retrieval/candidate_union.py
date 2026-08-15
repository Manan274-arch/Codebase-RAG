"""Deterministic, score-free union of BM25 and dense candidate pools."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.documents import Document

from src.retrieval.contracts import RankedDocument, canonical_chunk_id

CandidateProvenance = Literal["bm25", "dense", "both"]


class CandidateBranchRetriever(Protocol):
    """One independently ranked candidate source."""

    def retrieve(self, query: str, k: int = 10) -> Sequence[RankedDocument]: ...


@dataclass(frozen=True, slots=True)
class CandidateUnionResult:
    """An original document with branch membership and original branch ranks."""

    document: Document
    provenance: CandidateProvenance
    bm25_rank: int | None
    dense_rank: int | None


class CandidateUnionRetriever:
    """Combine bounded BM25 and dense pools without assigning a fused score.

    Output order is BM25 order followed by previously unseen dense results. A chunk
    found by both branches stays at its BM25 position and is marked ``both``.
    """

    def __init__(
        self,
        bm25_retriever: CandidateBranchRetriever,
        dense_retriever: CandidateBranchRetriever,
        *,
        bm25_depth: int = 25,
        dense_depth: int = 25,
    ) -> None:
        _validate_positive_integer(bm25_depth, "bm25_depth")
        _validate_positive_integer(dense_depth, "dense_depth")
        self._bm25_retriever = bm25_retriever
        self._dense_retriever = dense_retriever
        self._bm25_depth = bm25_depth
        self._dense_depth = dense_depth

    def retrieve(self, query: str, k: int = 10) -> list[CandidateUnionResult]:
        """Return up to ``k`` unique candidates with deterministic provenance."""
        _validate_non_negative_integer(k, "k")
        if k == 0 or not query.strip():
            return []
        bm25 = self._unique_branch(
            self._bm25_retriever.retrieve(query, k=self._bm25_depth), "BM25"
        )
        dense = self._unique_branch(
            self._dense_retriever.retrieve(query, k=self._dense_depth), "Dense"
        )
        dense_ranks = {
            canonical_chunk_id(item.document): rank
            for rank, item in enumerate(dense, start=1)
        }
        output = [
            CandidateUnionResult(
                document=item.document,
                provenance="both"
                if canonical_chunk_id(item.document) in dense_ranks
                else "bm25",
                bm25_rank=rank,
                dense_rank=dense_ranks.get(canonical_chunk_id(item.document)),
            )
            for rank, item in enumerate(bm25, start=1)
        ]
        seen = {canonical_chunk_id(item.document) for item in output}
        for dense_rank, item in enumerate(dense, start=1):
            chunk_id = canonical_chunk_id(item.document)
            if chunk_id not in seen:
                output.append(
                    CandidateUnionResult(item.document, "dense", None, dense_rank)
                )
                seen.add(chunk_id)
        return output[:k]

    @staticmethod
    def _unique_branch(
        results: Sequence[RankedDocument], branch: str
    ) -> list[RankedDocument]:
        unique: list[RankedDocument] = []
        seen: set[str] = set()
        for result in results:
            chunk_id = canonical_chunk_id(result.document)
            if chunk_id in seen:
                raise ValueError(
                    f"{branch} returned duplicate chunk identity: {chunk_id}"
                )
            seen.add(chunk_id)
            unique.append(result)
        return unique


def _validate_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
