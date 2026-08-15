"""Bounded post-reranking expansion of explicit HTTP relationships."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.documents import Document

from src.enrichment.relationships import (
    RELATED_HTTP_CALL_CHUNKS_KEY,
    RELATED_ROUTE_CHUNKS_KEY,
)
from src.retrieval.contracts import canonical_chunk_id
from src.retrieval.reranker import RerankedSearchResult

DEFAULT_EXPANSION_CANDIDATE_DEPTH = 50
DEFAULT_MAX_SEED_RESULTS = 5
DEFAULT_MAX_EXPANSIONS_PER_SEED = 2

ResultOrigin = Literal["retrieved", "relationship"]
RelationshipType = Literal["route", "http_call"]


class RerankedRetriever(Protocol):
    """Reranked candidate interface consumed by relationship expansion."""

    def retrieve(self, query: str, k: int = 10) -> Sequence[RerankedSearchResult]: ...


class RelationshipExpansionError(ValueError):
    """Raised when corpus identity or relationship metadata is inconsistent."""


@dataclass(frozen=True, slots=True)
class ExpandedSearchResult:
    """A retrieved or relationship-added Document with explicit provenance."""

    document: Document
    origin: ResultOrigin
    original_rank: int | None
    rerank_score: float | None
    first_stage_score: float | None
    first_stage_rank: int | None
    expanded_from_rank: int | None = None
    relationship_type: RelationshipType | None = None


class RelationshipExpander:
    """Insert direct linked chunks after bounded, reranked seed results."""

    def __init__(
        self,
        retriever: RerankedRetriever,
        corpus: Sequence[Document],
        *,
        candidate_depth: int = DEFAULT_EXPANSION_CANDIDATE_DEPTH,
        max_seed_results: int = DEFAULT_MAX_SEED_RESULTS,
        max_expansions_per_seed: int = DEFAULT_MAX_EXPANSIONS_PER_SEED,
    ) -> None:
        _validate_positive_integer(candidate_depth, "candidate_depth")
        _validate_non_negative_integer(max_seed_results, "max_seed_results")
        _validate_non_negative_integer(
            max_expansions_per_seed, "max_expansions_per_seed"
        )
        corpus_by_id: dict[str, Document] = {}
        for document in corpus:
            chunk_id = canonical_chunk_id(document)
            if chunk_id in corpus_by_id:
                raise RelationshipExpansionError(
                    f"corpus chunk identities must be unique: {chunk_id}"
                )
            corpus_by_id[chunk_id] = document
        self._retriever = retriever
        self._corpus_by_id = corpus_by_id
        self._candidate_depth = candidate_depth
        self._max_seed_results = max_seed_results
        self._max_expansions_per_seed = max_expansions_per_seed

    def retrieve(self, query: str, k: int = 10) -> list[ExpandedSearchResult]:
        """Return reranked seeds with direct relationship context inserted."""
        _validate_non_negative_integer(k, "k")
        if k == 0 or not query.strip():
            return []
        raw_seeds = self._retriever.retrieve(query, k=self._candidate_depth)
        seeds = _unique_seeds(raw_seeds)
        if not seeds:
            return []

        retrieved_by_id = {
            canonical_chunk_id(seed.document): (rank, seed)
            for rank, seed in enumerate(seeds, start=1)
        }
        output: list[ExpandedSearchResult] = []
        emitted: set[str] = set()

        for seed_rank, seed in enumerate(seeds, start=1):
            seed_id = canonical_chunk_id(seed.document)
            if seed_id not in emitted:
                output.append(_retrieved_result(seed, seed_rank))
                emitted.add(seed_id)
                if len(output) == k:
                    break

            if seed_rank > self._max_seed_results:
                continue
            added = 0
            for target_id, relationship_type in _relationship_targets(seed.document):
                if added == self._max_expansions_per_seed:
                    break
                if target_id in emitted:
                    continue
                target = self._corpus_by_id.get(target_id)
                if target is None:
                    raise RelationshipExpansionError(
                        f"relationship target is missing from corpus: {target_id}"
                    )
                retrieved = retrieved_by_id.get(target_id)
                if retrieved is None:
                    expanded = ExpandedSearchResult(
                        document=target,
                        origin="relationship",
                        original_rank=None,
                        rerank_score=None,
                        first_stage_score=None,
                        first_stage_rank=None,
                        expanded_from_rank=seed_rank,
                        relationship_type=relationship_type,
                    )
                else:
                    original_rank, retrieved_seed = retrieved
                    expanded = _retrieved_result(
                        retrieved_seed,
                        original_rank,
                        expanded_from_rank=seed_rank,
                        relationship_type=relationship_type,
                    )
                output.append(expanded)
                emitted.add(target_id)
                added += 1
                if len(output) == k:
                    break
            if len(output) == k:
                break

        return output


def _unique_seeds(
    seeds: Sequence[RerankedSearchResult],
) -> list[RerankedSearchResult]:
    unique: list[RerankedSearchResult] = []
    seen: set[str] = set()
    for seed in seeds:
        chunk_id = canonical_chunk_id(seed.document)
        if chunk_id not in seen:
            unique.append(seed)
            seen.add(chunk_id)
    return unique


def _retrieved_result(
    seed: RerankedSearchResult,
    original_rank: int,
    *,
    expanded_from_rank: int | None = None,
    relationship_type: RelationshipType | None = None,
) -> ExpandedSearchResult:
    return ExpandedSearchResult(
        document=seed.document,
        origin="retrieved",
        original_rank=original_rank,
        rerank_score=seed.score,
        first_stage_score=seed.first_stage_score,
        first_stage_rank=seed.first_stage_rank,
        expanded_from_rank=expanded_from_rank,
        relationship_type=relationship_type,
    )


def _relationship_targets(
    document: Document,
) -> list[tuple[str, RelationshipType]]:
    targets: list[tuple[str, RelationshipType]] = []
    relationship_keys: tuple[tuple[str, RelationshipType], ...] = (
        (RELATED_ROUTE_CHUNKS_KEY, "route"),
        (RELATED_HTTP_CALL_CHUNKS_KEY, "http_call"),
    )
    for key, relationship_type in relationship_keys:
        value = document.metadata.get(key, [])
        if not isinstance(value, list):
            raise RelationshipExpansionError(f"{key} metadata must be a list")
        for reference in value:
            targets.append((_reference_id(reference, key), relationship_type))
    return targets


def _reference_id(reference: object, key: str) -> str:
    if not isinstance(reference, Mapping):
        raise RelationshipExpansionError(f"{key} entries must be mappings")
    source = reference.get("source")
    chunk_index = reference.get("chunk_index")
    if not isinstance(source, str) or not source:
        raise RelationshipExpansionError(
            f"{key} entry requires a non-empty string source"
        )
    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or chunk_index < 0
    ):
        raise RelationshipExpansionError(
            f"{key} entry requires a non-negative integer chunk_index"
        )
    return f"{source}::{chunk_index}"


def _validate_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
