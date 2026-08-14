"""Retriever-agnostic offline evaluation for ranked LangChain documents."""

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

DEFAULT_KS = (1, 3, 5, 10)


class RankedDocument(Protocol):
    """A future-compatible ranked result containing a Document."""

    @property
    def document(self) -> Document: ...


class DocumentRetriever(Protocol):
    """The ranked-document behavior required by the evaluator."""

    def retrieve(self, query: str, k: int = 10) -> Sequence[RankedDocument]: ...


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationExample:
    query_id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    category: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    query: str
    category: str | None
    retrieved_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    first_relevant_rank: int | None
    metrics: Mapping[int, RetrievalMetrics]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    query_count: int
    ks: tuple[int, ...]
    metrics: Mapping[int, RetrievalMetrics]
    queries: tuple[QueryEvaluation, ...]
    category_metrics: Mapping[str, Mapping[int, RetrievalMetrics]]


class RetrievalEvaluationError(ValueError):
    """Raised for malformed benchmarks or incompatible retrieval results."""


def canonical_chunk_id(document: Document) -> str:
    """Encode the established ``source + chunk_index`` chunk identity."""
    source = document.metadata.get("source")
    chunk_index = document.metadata.get("chunk_index")
    if not isinstance(source, str) or not source:
        raise RetrievalEvaluationError(
            "chunk identity requires non-empty string 'source' metadata"
        )
    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or chunk_index < 0
    ):
        raise RetrievalEvaluationError(
            f"chunk {source!r} requires non-negative integer 'chunk_index' metadata"
        )
    return f"{source}::{chunk_index}"


def load_evaluation_examples(path: Path) -> tuple[RetrievalEvaluationExample, ...]:
    """Load and validate a human-readable JSON evaluation benchmark."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalEvaluationError(
            f"cannot load benchmark {path}: {error}"
        ) from error
    if not isinstance(raw, list):
        raise RetrievalEvaluationError("benchmark root must be a JSON list")

    examples: list[RetrievalEvaluationExample] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RetrievalEvaluationError(f"benchmark entry {index} must be an object")
        examples.append(_parse_example(item, index))
    _validate_examples(examples)
    return tuple(examples)


def evaluate_retriever(
    retriever: DocumentRetriever,
    corpus: Sequence[Document],
    examples: Sequence[RetrievalEvaluationExample],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalEvaluationResult:
    """Evaluate ranked Documents without inspecting retriever-specific scores."""
    if not examples:
        raise RetrievalEvaluationError("evaluation examples must not be empty")
    _validate_examples(examples)
    normalized_ks = _validate_ks(ks)
    corpus_ids = tuple(canonical_chunk_id(document) for document in corpus)
    if len(set(corpus_ids)) != len(corpus_ids):
        raise RetrievalEvaluationError("corpus chunk identities must be unique")
    corpus_id_set = frozenset(corpus_ids)
    missing = sorted(
        relevant_id
        for example in examples
        for relevant_id in example.relevant_chunk_ids
        if relevant_id not in corpus_id_set
    )
    if missing:
        missing_text = ", ".join(sorted(set(missing)))
        raise RetrievalEvaluationError(
            f"relevant chunk IDs are missing from corpus: {missing_text}"
        )

    query_results: list[QueryEvaluation] = []
    max_k = max(normalized_ks)
    for example in examples:
        ranked = retriever.retrieve(example.query, k=max_k)
        retrieved_ids = _unique_retrieved_ids(ranked, corpus_id_set)
        metrics = {
            k: _query_metrics(retrieved_ids, example.relevant_chunk_ids, k)
            for k in normalized_ks
        }
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in example.relevant_chunk_ids
            ),
            None,
        )
        query_results.append(
            QueryEvaluation(
                query_id=example.query_id,
                query=example.query,
                category=example.category,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=example.relevant_chunk_ids,
                first_relevant_rank=first_rank,
                metrics=metrics,
            )
        )

    aggregate = _mean_metrics(query_results, normalized_ks)
    categories: dict[str, list[QueryEvaluation]] = defaultdict(list)
    for result in query_results:
        if result.category is not None:
            categories[result.category].append(result)
    category_metrics = {
        category: _mean_metrics(results, normalized_ks)
        for category, results in sorted(categories.items())
    }
    return RetrievalEvaluationResult(
        query_count=len(query_results),
        ks=normalized_ks,
        metrics=aggregate,
        queries=tuple(query_results),
        category_metrics=category_metrics,
    )


def _parse_example(item: dict[str, Any], index: int) -> RetrievalEvaluationExample:
    query_id = item.get("query_id")
    query = item.get("query")
    relevant = item.get("relevant_chunk_ids")
    category = item.get("category")
    if not isinstance(query_id, str) or not query_id:
        raise RetrievalEvaluationError(f"benchmark entry {index} has invalid query_id")
    if not isinstance(query, str) or not query.strip():
        raise RetrievalEvaluationError(f"benchmark entry {index} has invalid query")
    if (
        not isinstance(relevant, list)
        or not relevant
        or not all(isinstance(value, str) and value for value in relevant)
    ):
        raise RetrievalEvaluationError(
            f"benchmark entry {index} requires non-empty relevant_chunk_ids"
        )
    if category is not None and not isinstance(category, str):
        raise RetrievalEvaluationError(f"benchmark entry {index} has invalid category")
    return RetrievalEvaluationExample(
        query_id=query_id,
        query=query,
        relevant_chunk_ids=frozenset(relevant),
        category=category,
    )


def _validate_examples(examples: Sequence[RetrievalEvaluationExample]) -> None:
    seen: set[str] = set()
    for example in examples:
        if not example.query_id:
            raise RetrievalEvaluationError("query_id must not be empty")
        if example.query_id in seen:
            raise RetrievalEvaluationError(f"duplicate query_id: {example.query_id}")
        if not example.query.strip():
            raise RetrievalEvaluationError(
                f"query {example.query_id!r} must not be empty"
            )
        if not example.relevant_chunk_ids:
            raise RetrievalEvaluationError(
                f"query {example.query_id!r} requires relevant chunks"
            )
        seen.add(example.query_id)


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    if not ks:
        raise RetrievalEvaluationError("at least one cutoff is required")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in ks):
        raise RetrievalEvaluationError("cutoffs must be positive integers")
    return tuple(sorted(set(ks)))


def _unique_retrieved_ids(
    ranked: Sequence[RankedDocument], corpus_ids: frozenset[str]
) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for result in ranked:
        chunk_id = canonical_chunk_id(result.document)
        if chunk_id not in corpus_ids:
            raise RetrievalEvaluationError(
                f"retriever returned chunk outside evaluated corpus: {chunk_id}"
            )
        if chunk_id not in seen:
            ids.append(chunk_id)
            seen.add(chunk_id)
    return tuple(ids)


def _query_metrics(
    retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int
) -> RetrievalMetrics:
    top_k = retrieved_ids[:k]
    relevance = [chunk_id in relevant_ids for chunk_id in top_k]
    relevant_count = sum(relevance)
    first_rank = next(
        (rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance, start=1)
        if is_relevant
    )
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return RetrievalMetrics(
        hit_rate=float(relevant_count > 0),
        recall=relevant_count / len(relevant_ids),
        mrr=0.0 if first_rank is None else 1.0 / first_rank,
        ndcg=0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg,
    )


def _mean_metrics(
    results: Sequence[QueryEvaluation], ks: tuple[int, ...]
) -> dict[int, RetrievalMetrics]:
    count = len(results)
    return {
        k: RetrievalMetrics(
            hit_rate=sum(result.metrics[k].hit_rate for result in results) / count,
            recall=sum(result.metrics[k].recall for result in results) / count,
            mrr=sum(result.metrics[k].mrr for result in results) / count,
            ndcg=sum(result.metrics[k].ndcg for result in results) / count,
        )
        for k in ks
    }
