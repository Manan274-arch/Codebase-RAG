"""Contracts and diagnostics for the frozen Retrieval Benchmark v2."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from langchain_core.documents import Document

from src.evaluation.metrics import (
    RetrievalEvaluationError,
    RetrievalEvaluationExample,
    RetrievalEvaluationResult,
    load_evaluation_examples,
)
from src.ingestion.relationships import (
    RELATED_HTTP_CALL_CHUNKS_KEY,
    RELATED_ROUTE_CHUNKS_KEY,
)
from src.retrieval.contracts import canonical_chunk_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_V2_REPOSITORY = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_repo_v2"
BENCHMARK_V2_QUERIES = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_v2.json"
BENCHMARK_V2_CATEGORIES = frozenset(
    {"lexical", "semantic", "structural", "relationship", "hard"}
)


def load_and_validate_benchmark_v2(
    corpus: Sequence[Document],
    benchmark: Path = BENCHMARK_V2_QUERIES,
) -> tuple[RetrievalEvaluationExample, ...]:
    """Load v2 and enforce its splits, categories, grades, and corpus identities."""
    examples = load_evaluation_examples(benchmark)
    if not 80 <= len(examples) <= 100:
        raise RetrievalEvaluationError("Benchmark v2 requires 80-100 queries")
    corpus_ids = frozenset(canonical_chunk_id(document) for document in corpus)
    if len(corpus_ids) != len(corpus):
        raise RetrievalEvaluationError("Benchmark v2 corpus identities must be unique")
    splits = {example.split for example in examples}
    if splits != {"dev", "test"}:
        raise RetrievalEvaluationError("Benchmark v2 requires dev and test splits")
    categories = {example.category for example in examples}
    if categories != BENCHMARK_V2_CATEGORIES:
        raise RetrievalEvaluationError("Benchmark v2 category coverage is incomplete")
    test_categories = {
        example.category for example in examples if example.split == "test"
    }
    if test_categories != BENCHMARK_V2_CATEGORIES:
        raise RetrievalEvaluationError(
            "Benchmark v2 test category coverage is incomplete"
        )
    for example in examples:
        if example.relevance_grades is None:
            raise RetrievalEvaluationError(
                f"Benchmark v2 query {example.query_id!r} requires graded relevance"
            )
        for chunk_id in example.relevant_chunk_ids:
            _validate_canonical_label(chunk_id, example.query_id)
            if chunk_id not in corpus_ids:
                raise RetrievalEvaluationError(
                    f"Benchmark v2 label is missing from corpus: {chunk_id}"
                )
    return examples


def linked_context_coverage(
    result: RetrievalEvaluationResult,
    examples: Sequence[RetrievalEvaluationExample],
    corpus: Sequence[Document],
    *,
    k: int,
) -> tuple[float, int]:
    """Measure whether an explicitly linked relevant pair is together in top-k.

    Eligible queries have at least one direct route/call edge whose two endpoints are
    both relevance-labeled. Coverage is the fraction with any such pair in top-k.
    """
    if k <= 0:
        raise RetrievalEvaluationError("linked context cutoff must be positive")
    documents = {canonical_chunk_id(document): document for document in corpus}
    examples_by_id = {example.query_id: example for example in examples}
    covered = 0
    eligible = 0
    for query_result in result.queries:
        example = examples_by_id[query_result.query_id]
        pairs = _relevant_linked_pairs(example, documents)
        if not pairs:
            continue
        eligible += 1
        retrieved = frozenset(query_result.retrieved_chunk_ids[:k])
        if any(first in retrieved and second in retrieved for first, second in pairs):
            covered += 1
    return (covered / eligible if eligible else 0.0), eligible


def _relevant_linked_pairs(
    example: RetrievalEvaluationExample, documents: Mapping[str, Document]
) -> frozenset[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    relevant = example.relevant_chunk_ids
    for source_id in sorted(relevant):
        document = documents[source_id]
        for target_id in _linked_ids(document):
            if target_id in relevant:
                first, second = sorted((source_id, target_id))
                pairs.add((first, second))
    return frozenset(pairs)


def _linked_ids(document: Document) -> tuple[str, ...]:
    linked: list[str] = []
    for key in (RELATED_ROUTE_CHUNKS_KEY, RELATED_HTTP_CALL_CHUNKS_KEY):
        value = document.metadata.get(key, [])
        if not isinstance(value, list):
            continue
        for reference in value:
            if not isinstance(reference, Mapping):
                continue
            source = reference.get("source")
            chunk_index = reference.get("chunk_index")
            if (
                isinstance(source, str)
                and source
                and isinstance(chunk_index, int)
                and not isinstance(chunk_index, bool)
                and chunk_index >= 0
            ):
                linked.append(f"{source}::{chunk_index}")
    return tuple(linked)


def _validate_canonical_label(chunk_id: str, query_id: str) -> None:
    source, separator, index = chunk_id.rpartition("::")
    if not separator or not source or not index.isdigit():
        raise RetrievalEvaluationError(
            f"Benchmark v2 query {query_id!r} has malformed chunk ID: {chunk_id}"
        )
