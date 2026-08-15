"""Candidate-pool recall and provenance diagnostics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.evaluation.metrics import (
    DocumentRetriever,
    RetrievalEvaluationExample,
)
from src.retrieval.contracts import canonical_chunk_id


@dataclass(frozen=True, slots=True)
class CandidateRecall:
    """Macro relevant-chunk recall and query hit rate at one cutoff."""

    recall: float
    hit_rate: float


@dataclass(frozen=True, slots=True)
class CandidateProvenanceCounts:
    """Aggregate union membership counts across query pools."""

    bm25_only: int
    dense_only: int
    both: int


def evaluate_candidate_recall(
    retriever: DocumentRetriever,
    examples: Sequence[RetrievalEvaluationExample],
    *,
    ks: Sequence[int] = (10, 20, 50),
) -> Mapping[int, CandidateRecall]:
    """Measure macro fraction of labels present and queries with any label present."""
    if not examples:
        raise ValueError("candidate examples must not be empty")
    cutoffs = tuple(sorted(set(ks)))
    if not cutoffs or any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in cutoffs
    ):
        raise ValueError("candidate cutoffs must be positive integers")
    retrieved = {
        example.query_id: tuple(
            canonical_chunk_id(item.document)
            for item in retriever.retrieve(example.query, k=max(cutoffs))
        )
        for example in examples
    }
    output: dict[int, CandidateRecall] = {}
    for k in cutoffs:
        recalls: list[float] = []
        hits = 0
        for example in examples:
            present = frozenset(retrieved[example.query_id][:k])
            found = len(present & example.relevant_chunk_ids)
            recalls.append(found / len(example.relevant_chunk_ids))
            hits += found > 0
        output[k] = CandidateRecall(
            recall=sum(recalls) / len(recalls), hit_rate=hits / len(examples)
        )
    return output


def count_union_provenance(
    retriever: DocumentRetriever,
    examples: Sequence[RetrievalEvaluationExample],
    *,
    k: int = 50,
) -> CandidateProvenanceCounts:
    """Count query-candidate occurrences by union provenance."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("provenance cutoff must be a positive integer")
    counts = {"bm25": 0, "dense": 0, "both": 0}
    for example in examples:
        for result in retriever.retrieve(example.query, k=k):
            provenance = getattr(result, "provenance", None)
            if provenance not in counts:
                raise ValueError("candidate result has invalid provenance")
            counts[provenance] += 1
    return CandidateProvenanceCounts(counts["bm25"], counts["dense"], counts["both"])
