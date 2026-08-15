import math
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.documents import Document
from src.evaluation.metrics import (
    RetrievalEvaluationError,
    RetrievalEvaluationExample,
    canonical_chunk_id,
    evaluate_retriever,
    load_evaluation_examples,
)
from src.evaluation.runners.evaluate_bm25 import run_benchmark


def chunk(source: str, chunk_index: int) -> Document:
    return Document(
        page_content=f"content for {source}",
        metadata={"source": source, "chunk_index": chunk_index},
    )


@dataclass(frozen=True)
class FakeResult:
    document: Document


class FakeRetriever:
    def __init__(self, ranking: list[Document]) -> None:
        self.ranking = ranking

    def retrieve(self, query: str, k: int = 10) -> list[FakeResult]:
        del query
        return [FakeResult(document) for document in self.ranking[:k]]


def example(
    relevant: frozenset[str], *, query_id: str = "query"
) -> RetrievalEvaluationExample:
    return RetrievalEvaluationExample(
        query_id=query_id,
        query="natural language question",
        relevant_chunk_ids=relevant,
        category="test",
    )


def test_metrics_have_exact_hit_recall_mrr_and_binary_ndcg() -> None:
    corpus = [chunk("a.py", 0), chunk("b.py", 0), chunk("c.py", 0)]
    relevant = frozenset({"b.py::0", "c.py::0"})

    result = evaluate_retriever(
        FakeRetriever(corpus), corpus, [example(relevant)], ks=(1, 2, 3)
    )

    assert result.metrics[1].hit_rate == 0.0
    assert result.metrics[1].recall == 0.0
    assert result.metrics[1].mrr == 0.0
    assert result.metrics[1].ndcg == 0.0
    assert result.metrics[2].hit_rate == 1.0
    assert result.metrics[2].recall == 0.5
    assert result.metrics[2].mrr == 0.5
    expected_ndcg_at_2 = (1 / math.log2(3)) / (
        1 + 1 / math.log2(3)
    )
    assert result.metrics[2].ndcg == pytest.approx(expected_ndcg_at_2)
    assert result.metrics[3].recall == 1.0
    expected_ndcg_at_3 = (1 / math.log2(3) + 1 / math.log2(4)) / (
        1 + 1 / math.log2(3)
    )
    assert result.metrics[3].ndcg == pytest.approx(expected_ndcg_at_3)
    assert result.queries[0].first_relevant_rank == 2
    assert result.queries[0].retrieved_chunk_ids == (
        "a.py::0",
        "b.py::0",
        "c.py::0",
    )


@pytest.mark.parametrize(
    ("ranking", "k", "expected_mrr"),
    [
        (("b.py", "a.py", "c.py"), 3, 1.0),
        (("a.py", "b.py", "c.py"), 3, 0.5),
        (("a.py", "c.py", "b.py"), 2, 0.0),
    ],
)
def test_mrr_rank_one_later_and_outside_cutoff(
    ranking: tuple[str, ...], k: int, expected_mrr: float
) -> None:
    by_source = {source: chunk(source, 0) for source in ranking}
    corpus = list(by_source.values())
    ranked = [by_source[source] for source in ranking]

    result = evaluate_retriever(
        FakeRetriever(ranked),
        corpus,
        [example(frozenset({"b.py::0"}))],
        ks=(k,),
    )

    assert result.metrics[k].mrr == expected_mrr


def test_metrics_are_macro_averaged_and_category_diagnostics_are_retained() -> None:
    corpus = [chunk("a.py", 0), chunk("b.py", 0)]
    examples = [
        example(frozenset({"a.py::0"}), query_id="hit"),
        RetrievalEvaluationExample(
            query_id="miss",
            query="another question",
            relevant_chunk_ids=frozenset({"b.py::0"}),
            category="other",
        ),
    ]

    result = evaluate_retriever(
        FakeRetriever([corpus[0]]), corpus, examples, ks=(1,)
    )

    assert result.metrics[1].hit_rate == 0.5
    assert set(result.category_metrics) == {"other", "test"}
    assert result.query_count == 2


def test_missing_relevant_chunk_fails_loudly() -> None:
    corpus = [chunk("present.py", 0)]

    with pytest.raises(RetrievalEvaluationError, match="missing.py::0"):
        evaluate_retriever(
            FakeRetriever(corpus),
            corpus,
            [example(frozenset({"missing.py::0"}))],
        )


def test_duplicate_query_ids_are_rejected() -> None:
    corpus = [chunk("a.py", 0)]
    examples = [
        example(frozenset({"a.py::0"}), query_id="duplicate"),
        example(frozenset({"a.py::0"}), query_id="duplicate"),
    ]

    with pytest.raises(RetrievalEvaluationError, match="duplicate query_id"):
        evaluate_retriever(FakeRetriever(corpus), corpus, examples)


def test_empty_evaluation_set_is_rejected() -> None:
    with pytest.raises(RetrievalEvaluationError, match="must not be empty"):
        evaluate_retriever(FakeRetriever([]), [], [])


def test_canonical_identity_uses_existing_source_and_chunk_index() -> None:
    assert canonical_chunk_id(chunk("nested/file.py", 4)) == "nested/file.py::4"


def test_loader_validates_human_readable_json(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        '[{"query_id":"q","query":"question","relevant_chunk_ids":[]}]',
        encoding="utf-8",
    )

    with pytest.raises(RetrievalEvaluationError, match="relevant_chunk_ids"):
        load_evaluation_examples(benchmark)


def test_committed_bm25_benchmark_runs_through_real_ingestion() -> None:
    result = run_benchmark()

    assert result.query_count == 15
    assert result.ks == (1, 3, 5, 10)
    assert set(result.category_metrics) == {"lexical", "relationship", "semantic"}
    assert all(0.0 <= metrics.ndcg <= 1.0 for metrics in result.metrics.values())
