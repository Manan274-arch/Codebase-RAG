import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.documents import Document
from src.evaluation.benchmark_v2 import (
    BENCHMARK_V2_CATEGORIES,
    BENCHMARK_V2_QUERIES,
    BENCHMARK_V2_REPOSITORY,
    linked_context_coverage,
    load_and_validate_benchmark_v2,
)
from src.evaluation.metrics import (
    RetrievalEvaluationExample,
    evaluate_retriever,
    load_evaluation_examples,
    select_examples,
    subset_evaluation_result,
)
from src.ingestion.pipeline import build_enriched_corpus
from src.ingestion.repository import find_source_files


@dataclass(frozen=True)
class FakeResult:
    document: Document


class QueryRankingRetriever:
    def __init__(self, rankings: dict[str, list[Document]]) -> None:
        self.rankings = rankings

    def retrieve(self, query: str, k: int = 10) -> list[FakeResult]:
        return [FakeResult(item) for item in self.rankings[query][:k]]


@pytest.fixture(scope="module")
def v2_corpus() -> list[Document]:
    return build_enriched_corpus(BENCHMARK_V2_REPOSITORY)


@pytest.fixture(scope="module")
def v2_examples(
    v2_corpus: list[Document],
) -> tuple[RetrievalEvaluationExample, ...]:
    return load_and_validate_benchmark_v2(v2_corpus)


def test_v2_repository_ingests_through_production_pipeline(
    v2_corpus: list[Document],
) -> None:
    source_files = find_source_files(BENCHMARK_V2_REPOSITORY)

    assert 40 <= len(source_files) <= 70
    assert len(source_files) == 56
    assert len(v2_corpus) == 56
    assert {item.metadata["language"] for item in v2_corpus} == {
        "python",
        "typescript",
    }
    assert all("raw_content" in item.metadata for item in v2_corpus)


def test_v2_query_counts_splits_categories_and_ids_are_frozen(
    v2_examples: tuple[RetrievalEvaluationExample, ...],
) -> None:
    assert len(v2_examples) == 90
    assert len({item.query_id for item in v2_examples}) == 90
    assert Counter(item.split for item in v2_examples) == {"dev": 20, "test": 70}
    assert Counter(item.category for item in v2_examples) == {
        category: 18 for category in BENCHMARK_V2_CATEGORIES
    }
    for split, count in (("dev", 4), ("test", 14)):
        assert Counter(
            item.category for item in v2_examples if item.split == split
        ) == {category: count for category in BENCHMARK_V2_CATEGORIES}


def test_v2_has_valid_graded_and_multi_relevant_labels(
    v2_examples: tuple[RetrievalEvaluationExample, ...],
) -> None:
    assert all(item.relevance_grades is not None for item in v2_examples)
    assert all(
        set(item.relevance_grades.values()).issubset({1, 2})
        for item in v2_examples
        if item.relevance_grades is not None
    )
    assert sum(len(item.relevant_chunk_ids) == 1 for item in v2_examples) == 25
    assert sum(len(item.relevant_chunk_ids) > 1 for item in v2_examples) == 65


def test_split_selection_and_reaggregation_do_not_rerun_retrieval(
    v2_corpus: list[Document],
    v2_examples: tuple[RetrievalEvaluationExample, ...],
) -> None:
    sample = v2_examples[:5]
    rankings = {item.query: v2_corpus for item in sample}
    result = evaluate_retriever(
        QueryRankingRetriever(rankings), v2_corpus, sample, ks=(1, 3)
    )
    subset = subset_evaluation_result(
        result, frozenset(item.query_id for item in sample[:2])
    )

    assert subset.query_count == 2
    assert subset.ks == (1, 3)
    assert len(select_examples(v2_examples, "dev")) == 20
    assert len(select_examples(v2_examples, "test")) == 70


def test_graded_ndcg_rewards_primary_before_supporting() -> None:
    primary = Document(
        page_content="primary", metadata={"source": "primary.py", "chunk_index": 0}
    )
    support = Document(
        page_content="support", metadata={"source": "support.py", "chunk_index": 0}
    )
    example = RetrievalEvaluationExample(
        query_id="graded",
        query="question",
        relevant_chunk_ids=frozenset({"primary.py::0", "support.py::0"}),
        relevance_grades={"primary.py::0": 2, "support.py::0": 1},
    )

    ideal = evaluate_retriever(
        QueryRankingRetriever({"question": [primary, support]}),
        [primary, support],
        [example],
        ks=(2,),
    )
    reversed_result = evaluate_retriever(
        QueryRankingRetriever({"question": [support, primary]}),
        [primary, support],
        [example],
        ks=(2,),
    )

    assert ideal.metrics[2].ndcg == 1.0
    assert reversed_result.metrics[2].ndcg < 1.0
    assert reversed_result.metrics[2].hit_rate == 1.0
    assert reversed_result.metrics[2].recall == 1.0


def test_linked_context_coverage_is_additive_and_deterministic(
    v2_corpus: list[Document],
    v2_examples: tuple[RetrievalEvaluationExample, ...],
) -> None:
    example = next(item for item in v2_examples if item.query_id == "relationship_01")
    by_id = {
        f"{item.metadata['source']}::{item.metadata['chunk_index']}": item
        for item in v2_corpus
    }
    relevant = [by_id[chunk_id] for chunk_id in sorted(example.relevant_chunk_ids)]
    result = evaluate_retriever(
        QueryRankingRetriever({example.query: relevant}),
        v2_corpus,
        [example],
        ks=(1, 2),
    )

    assert linked_context_coverage(result, [example], v2_corpus, k=1) == (0.0, 1)
    assert linked_context_coverage(result, [example], v2_corpus, k=2) == (1.0, 1)


def test_v1_fixture_is_unchanged_and_backward_compatible() -> None:
    v1_path = Path("tests/fixtures/retrieval_eval.json")
    digest = hashlib.sha256(v1_path.read_bytes()).hexdigest()
    examples = load_evaluation_examples(v1_path)

    assert digest == "e8ddf95715450495a39b0ebad7316ffd6af58947298a6343f1dd5ae2d085bc02"
    assert len(examples) == 15
    assert all(item.relevance_grades is None for item in examples)
    assert all(item.split is None for item in examples)


def test_v2_json_is_deterministic_generated_content() -> None:
    first = BENCHMARK_V2_QUERIES.read_bytes()
    second = BENCHMARK_V2_QUERIES.read_bytes()

    assert first == second
    assert len(load_evaluation_examples(BENCHMARK_V2_QUERIES)) == 90
