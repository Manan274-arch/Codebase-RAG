from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from langchain_core.documents import Document
from src.evaluation.brute_force_dense import DenseRetriever, DenseSearchResult
from src.evaluation.metrics import (
    RetrievalEvaluationExample,
    canonical_chunk_id,
    evaluate_retriever,
)
from src.evaluation.rrf import (
    DEFAULT_RRF_CONSTANT,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from src.retrieval.bm25 import BM25Retriever, BM25SearchResult


def document(content: str, source: str, chunk_index: int) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "chunk_index": chunk_index, "marker": source},
    )


def result(item: Document, score: float) -> BM25SearchResult:
    return BM25SearchResult(document=item, score=score)


class FakeRetriever:
    def __init__(self, results: Sequence[BM25SearchResult | DenseSearchResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(
        self, query: str, k: int = 10
    ) -> list[BM25SearchResult | DenseSearchResult]:
        self.calls.append((query, k))
        return self.results[:k]


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> npt.NDArray[np.float32]:
        assert normalize_embeddings
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_standard_rrf_formula_uses_one_based_ranks_and_default_constant() -> None:
    first = document("first", "a.py", 0)
    shared = document("shared", "b.py", 0)
    last = document("last", "c.py", 0)

    fused = reciprocal_rank_fusion(
        (
            [result(first, 999.0), result(shared, 1.0)],
            [result(shared, -40.0), result(last, 10000.0)],
        )
    )

    assert DEFAULT_RRF_CONSTANT == 60
    assert [canonical_chunk_id(item.document) for item in fused] == [
        "b.py::0",
        "a.py::0",
        "c.py::0",
    ]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)


def test_raw_scores_never_change_fused_scores_or_order() -> None:
    first = document("first", "a.py", 0)
    second = document("second", "b.py", 0)
    rankings = ([first, second], [second, first])

    low = reciprocal_rank_fusion(
        tuple([result(item, -1.0) for item in branch] for branch in rankings)
    )
    high = reciprocal_rank_fusion(
        tuple([result(item, 1_000_000.0) for item in branch] for branch in rankings)
    )

    assert [(canonical_chunk_id(item.document), item.score) for item in low] == [
        (canonical_chunk_id(item.document), item.score) for item in high
    ]


def test_canonical_deduplication_preserves_first_original_document() -> None:
    original = document("source from bm25", "shared.py", 2)
    same_identity = document("different object", "shared.py", 2)

    fused = reciprocal_rank_fusion(
        ([result(original, 1.0)], [result(same_identity, 2.0)])
    )

    assert len(fused) == 1
    assert fused[0].document is original
    assert fused[0].document.metadata["marker"] == "shared.py"


def test_duplicate_within_one_branch_contributes_only_at_first_rank() -> None:
    item = document("code", "a.py", 0)

    fused = reciprocal_rank_fusion(([result(item, 1.0), result(item, 0.0)],))

    assert fused[0].score == pytest.approx(1 / 61)


def test_ties_use_best_rank_then_stable_first_appearance() -> None:
    first = document("first", "a.py", 0)
    second = document("second", "b.py", 0)
    third = document("third", "c.py", 0)
    ranked_lists = (
        [result(first, 0.0), result(second, 0.0)],
        [result(third, 0.0), result(second, 0.0)],
    )

    runs = [reciprocal_rank_fusion(ranked_lists) for _ in range(3)]

    assert [item.document for item in runs[0]] == [second, first, third]
    assert all(
        [canonical_chunk_id(item.document) for item in run]
        == ["b.py::0", "a.py::0", "c.py::0"]
        for run in runs
    )


def test_top_k_empty_lists_and_invalid_arguments() -> None:
    item = document("code", "a.py", 0)
    ranked = ([result(item, 0.0)],)

    assert reciprocal_rank_fusion(ranked, top_k=0) == []
    assert len(reciprocal_rank_fusion(ranked, top_k=20)) == 1
    assert reciprocal_rank_fusion(([], [])) == []
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion(ranked, top_k=-1)
    with pytest.raises(ValueError, match="rrf_constant"):
        reciprocal_rank_fusion(ranked, rrf_constant=0)


def test_malformed_canonical_identity_fails_consistently() -> None:
    malformed = Document(page_content="code", metadata={"source": "a.py"})

    with pytest.raises(ValueError, match="chunk_index"):
        reciprocal_rank_fusion(([result(malformed, 1.0)],))


def test_hybrid_uses_independent_candidate_depth_and_final_k() -> None:
    documents = [document(str(index), f"{index}.py", 0) for index in range(4)]
    bm25 = FakeRetriever([result(item, 0.0) for item in documents[:3]])
    dense = FakeRetriever([result(item, 0.0) for item in reversed(documents[1:])])
    hybrid = HybridRetriever(bm25, dense, candidate_depth=3)

    fused = hybrid.retrieve("query", k=2)

    assert len(fused) == 2
    assert bm25.calls == [("query", 3)]
    assert dense.calls == [("query", 3)]


def test_hybrid_edge_cases_and_short_branches() -> None:
    item = document("code", "a.py", 0)
    bm25 = FakeRetriever([result(item, 1.0)])
    dense = FakeRetriever([])
    hybrid = HybridRetriever(bm25, dense, candidate_depth=10)

    assert hybrid.retrieve("query", k=20)[0].document is item
    calls_before = (list(bm25.calls), list(dense.calls))
    assert hybrid.retrieve(" ") == []
    assert hybrid.retrieve("query", k=0) == []
    assert (bm25.calls, dense.calls) == calls_before
    with pytest.raises(ValueError, match="k"):
        hybrid.retrieve("query", k=-1)
    with pytest.raises(ValueError, match="candidate_depth"):
        HybridRetriever(bm25, dense, candidate_depth=0)


def test_actual_bm25_and_dense_interfaces_integrate_with_evaluator() -> None:
    documents = [
        document("validate authentication token", "auth.py", 0),
        document("render dashboard", "view.ts", 0),
    ]
    encoder = FakeEncoder(
        {
            "validate authentication token": [1.0, 0.0],
            "render dashboard": [0.0, 1.0],
            "where are credentials checked": [1.0, 0.0],
        }
    )
    hybrid = HybridRetriever(
        BM25Retriever(documents),
        DenseRetriever(documents, encoder=encoder),
        candidate_depth=10,
    )
    examples = [
        RetrievalEvaluationExample(
            query_id="auth",
            query="where are credentials checked",
            relevant_chunk_ids=frozenset({"auth.py::0"}),
        )
    ]

    evaluation = evaluate_retriever(hybrid, documents, examples, ks=(1, 3))

    assert evaluation.metrics[1].hit_rate == 1.0
    assert evaluation.queries[0].retrieved_chunk_ids[0] == "auth.py::0"


def test_empty_actual_retrievers_produce_no_hybrid_results() -> None:
    encoder = FakeEncoder({})
    hybrid = HybridRetriever(BM25Retriever([]), DenseRetriever([], encoder=encoder))

    assert hybrid.retrieve("query") == []
