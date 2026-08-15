from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from langchain_core.documents import Document
from src.evaluation.candidate_diagnostics import (
    count_union_provenance,
    evaluate_candidate_recall,
)
from src.evaluation.metrics import RetrievalEvaluationExample, evaluate_retriever
from src.retrieval.candidate_union import CandidateUnionRetriever
from src.retrieval.reranker import CrossEncoderReranker


@dataclass(frozen=True)
class Result:
    document: Document
    score: float = 1.0


class FakeRetriever:
    def __init__(self, results: Sequence[Result]) -> None:
        self.results = list(results)
        self.calls: list[int] = []

    def retrieve(self, query: str, k: int = 10) -> list[Result]:
        self.calls.append(k)
        return self.results[:k]


class Scorer:
    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return np.asarray([float(pair[1]) for pair in pairs], dtype=np.float32)


def doc(name: str, value: str = "1") -> Document:
    return Document(
        page_content=value, metadata={"source": f"{name}.py", "chunk_index": 0}
    )


def test_union_deduplicates_and_preserves_deterministic_provenance() -> None:
    a, b, c, d = (doc(name) for name in "abcd")
    union = CandidateUnionRetriever(
        FakeRetriever([Result(a), Result(b), Result(c)]),
        FakeRetriever([Result(c), Result(a), Result(d)]),
        bm25_depth=3,
        dense_depth=3,
    )
    results = union.retrieve("query", k=10)
    assert [item.document for item in results] == [a, b, c, d]
    assert [item.provenance for item in results] == ["both", "bm25", "both", "dense"]
    assert [(item.bm25_rank, item.dense_rank) for item in results] == [
        (1, 2),
        (2, None),
        (3, 1),
        (None, 3),
    ]
    assert results[0].document is a


def test_union_depths_k_empty_branches_and_no_mutation() -> None:
    a, b, c = doc("a"), doc("b"), doc("c")
    original = dict(a.metadata)
    bm25, dense = FakeRetriever([Result(a), Result(b)]), FakeRetriever([Result(c)])
    union = CandidateUnionRetriever(bm25, dense, bm25_depth=1, dense_depth=1)
    assert union.retrieve("query", k=0) == []
    assert [item.document for item in union.retrieve("query", k=1)] == [a]
    assert bm25.calls == [1] and dense.calls == [1]
    assert a.metadata == original
    assert (
        CandidateUnionRetriever(FakeRetriever([]), dense).retrieve("q")[0].provenance
        == "dense"
    )
    assert (
        CandidateUnionRetriever(bm25, FakeRetriever([])).retrieve("q")[0].provenance
        == "bm25"
    )
    assert (
        CandidateUnionRetriever(FakeRetriever([]), FakeRetriever([])).retrieve("q")
        == []
    )


@pytest.mark.parametrize("name,value", [("bm25_depth", 0), ("dense_depth", -1)])
def test_union_rejects_invalid_depths(name: str, value: int) -> None:
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        CandidateUnionRetriever(FakeRetriever([]), FakeRetriever([]), **kwargs)


def test_union_rejects_duplicate_canonical_identity_in_a_branch() -> None:
    a = doc("a")
    union = CandidateUnionRetriever(
        FakeRetriever([Result(a), Result(a)]), FakeRetriever([])
    )
    with pytest.raises(ValueError, match="duplicate chunk identity"):
        union.retrieve("q")


def test_union_is_cross_encoder_and_evaluator_compatible() -> None:
    a, b = doc("a", "1"), doc("b", "2")
    union = CandidateUnionRetriever(
        FakeRetriever([Result(a)]), FakeRetriever([Result(b)])
    )
    reranker = CrossEncoderReranker(union, scorer=Scorer())
    ranked = reranker.retrieve("q", k=2)
    assert [item.document for item in ranked] == [b, a]
    assert ranked[0].candidate_provenance == "dense"
    assert ranked[0].first_stage_score is None
    example = RetrievalEvaluationExample("q", "q", frozenset({"b.py::0"}))
    assert (
        evaluate_retriever(reranker, [a, b], [example], ks=(1,)).metrics[1].hit_rate
        == 1.0
    )


def test_candidate_recall_and_provenance_metrics() -> None:
    a, b, c = doc("a"), doc("b"), doc("c")
    union = CandidateUnionRetriever(
        FakeRetriever([Result(a), Result(b)]), FakeRetriever([Result(b), Result(c)])
    )
    examples = (
        RetrievalEvaluationExample("q", "q", frozenset({"a.py::0", "c.py::0"})),
    )
    metrics = evaluate_candidate_recall(union, examples, ks=(1, 3))
    assert metrics[1].recall == 0.5 and metrics[1].hit_rate == 1.0
    assert metrics[3].recall == 1.0 and metrics[3].hit_rate == 1.0
    counts = count_union_provenance(union, examples, k=3)
    assert (counts.bm25_only, counts.dense_only, counts.both) == (1, 1, 1)
