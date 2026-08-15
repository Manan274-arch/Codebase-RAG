from collections.abc import Sequence
from typing import Any

import pytest
from evaluation.metrics import (
    RetrievalEvaluationExample,
    canonical_chunk_id,
    evaluate_retriever,
)
from langchain_core.documents import Document
from src.retrieval.relationship_expansion import (
    RelationshipExpander,
    RelationshipExpansionError,
)
from src.retrieval.reranker import RerankedSearchResult


def reference(source: str, chunk_index: int = 0) -> dict[str, object]:
    return {
        "source": source,
        "chunk_index": chunk_index,
        "start_line": 1,
        "end_line": 2,
    }


def document(
    name: str,
    *,
    routes: Sequence[str] = (),
    calls: Sequence[str] = (),
) -> Document:
    return Document(
        page_content=f"Source: {name}.py\n\nCode:\n{name}()",
        metadata={
            "source": f"{name}.py",
            "chunk_index": 0,
            "raw_content": f"{name}()",
            "related_route_chunks": [reference(item) for item in routes],
            "related_http_call_chunks": [reference(item) for item in calls],
        },
    )


def reranked(item: Document, rank: int) -> RerankedSearchResult:
    return RerankedSearchResult(
        document=item,
        score=1.0 / rank,
        first_stage_score=1.0 / (60 + rank),
        first_stage_rank=rank,
    )


class FakeReranker:
    def __init__(self, results: Sequence[RerankedSearchResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 10) -> list[RerankedSearchResult]:
        self.calls.append((query, k))
        return self.results[:k]


def ids(results: Sequence[Any]) -> list[str]:
    return [canonical_chunk_id(item.document) for item in results]


def test_call_seed_inserts_related_route_immediately_after_seed() -> None:
    route = document("backend")
    call = document("frontend", routes=("backend.py",))
    other = document("other")
    retriever = FakeReranker([reranked(call, 1), reranked(other, 2)])

    results = RelationshipExpander(retriever, [call, route, other]).retrieve("query")

    assert ids(results) == ["frontend.py::0", "backend.py::0", "other.py::0"]
    assert results[0].origin == "retrieved"
    assert results[1].origin == "relationship"
    assert results[1].relationship_type == "route"
    assert results[1].expanded_from_rank == 1
    assert results[1].rerank_score is None
    assert results[1].first_stage_score is None


def test_route_seed_expands_reverse_http_call_relationship() -> None:
    call = document("frontend")
    route = document("backend", calls=("frontend.py",))

    results = RelationshipExpander(
        FakeReranker([reranked(route, 1)]), [route, call]
    ).retrieve("query")

    assert ids(results) == ["backend.py::0", "frontend.py::0"]
    assert results[1].relationship_type == "http_call"


def test_already_reranked_target_is_promoted_but_keeps_real_scores() -> None:
    target = document("target")
    seed = document("seed", routes=("target.py",))
    other = document("other")
    target_result = reranked(target, 3)
    retriever = FakeReranker(
        [reranked(seed, 1), reranked(other, 2), target_result]
    )

    results = RelationshipExpander(retriever, [seed, other, target]).retrieve("query")

    assert ids(results) == ["seed.py::0", "target.py::0", "other.py::0"]
    assert results[1].origin == "retrieved"
    assert results[1].original_rank == 3
    assert results[1].rerank_score == target_result.score
    assert results[1].first_stage_score == target_result.first_stage_score
    assert results[1].first_stage_rank == target_result.first_stage_rank


def test_shared_target_and_duplicate_normal_seed_appear_once() -> None:
    target = document("target")
    first = document("first", routes=("target.py",))
    second = document("second", routes=("target.py",))
    retriever = FakeReranker(
        [reranked(first, 1), reranked(second, 2), reranked(target, 3)]
    )

    results = RelationshipExpander(retriever, [first, second, target]).retrieve("q")

    assert ids(results) == ["first.py::0", "target.py::0", "second.py::0"]


def test_seed_expansion_and_final_result_limits_are_independent() -> None:
    one = document("one")
    two = document("two")
    three = document("three")
    first = document("first", routes=("one.py", "two.py"))
    second = document("second", routes=("three.py",))
    retriever = FakeReranker([reranked(first, 1), reranked(second, 2)])
    expander = RelationshipExpander(
        retriever,
        [first, second, one, two, three],
        candidate_depth=8,
        max_seed_results=1,
        max_expansions_per_seed=1,
    )

    results = expander.retrieve("query", k=2)

    assert retriever.calls == [("query", 8)]
    assert ids(results) == ["first.py::0", "one.py::0"]


def test_no_relationships_preserve_normal_ranking_and_edge_cases() -> None:
    first = document("first")
    second = document("second")
    retriever = FakeReranker([reranked(first, 1), reranked(second, 2)])
    expander = RelationshipExpander(retriever, [first, second])

    assert ids(expander.retrieve("query")) == ["first.py::0", "second.py::0"]
    calls_before = list(retriever.calls)
    assert expander.retrieve("query", k=0) == []
    assert expander.retrieve(" ") == []
    assert retriever.calls == calls_before
    with pytest.raises(ValueError, match="k"):
        expander.retrieve("query", k=-1)

    empty = RelationshipExpander(FakeReranker([]), [])
    assert empty.retrieve("query") == []


def test_relationship_order_is_stable_and_respects_metadata_order() -> None:
    first_target = document("a")
    second_target = document("b")
    seed = document("seed", routes=("b.py", "a.py"))
    expander = RelationshipExpander(
        FakeReranker([reranked(seed, 1)]),
        [seed, first_target, second_target],
    )

    runs = [expander.retrieve("query") for _ in range(3)]

    assert all(ids(run) == ["seed.py::0", "b.py::0", "a.py::0"] for run in runs)


def test_expansion_is_non_recursive_when_added_target_has_its_own_link() -> None:
    third = document("third")
    second = document("second", routes=("third.py",))
    first = document("first", routes=("second.py",))

    results = RelationshipExpander(
        FakeReranker([reranked(first, 1)]), [first, second, third]
    ).retrieve("query")

    assert ids(results) == ["first.py::0", "second.py::0"]


def test_documents_identity_metadata_and_content_are_not_mutated() -> None:
    target = document("target")
    seed = document("seed", routes=("target.py",))
    corpus = [seed, target]
    metadata_before = [dict(item.metadata) for item in corpus]
    content_before = [item.page_content for item in corpus]

    results = RelationshipExpander(
        FakeReranker([reranked(seed, 1)]), corpus
    ).retrieve("query")

    assert results[0].document is seed
    assert results[1].document is target
    assert [item.metadata for item in corpus] == metadata_before
    assert [item.page_content for item in corpus] == content_before


def test_generic_evaluator_accepts_expanded_results() -> None:
    target = document("target")
    seed = document("seed", routes=("target.py",))
    expander = RelationshipExpander(
        FakeReranker([reranked(seed, 1)]), [seed, target]
    )
    examples = [
        RetrievalEvaluationExample(
            query_id="target",
            query="find linked target",
            relevant_chunk_ids=frozenset({"target.py::0"}),
        )
    ]

    evaluation = evaluate_retriever(expander, [seed, target], examples, ks=(1, 2))

    assert evaluation.metrics[1].hit_rate == 0.0
    assert evaluation.metrics[2].hit_rate == 1.0


def test_invalid_corpus_limits_and_relationship_metadata_fail_clearly() -> None:
    seed = document("seed", routes=("missing.py",))
    reranker = FakeReranker([reranked(seed, 1)])
    with pytest.raises(ValueError, match="candidate_depth"):
        RelationshipExpander(reranker, [seed], candidate_depth=0)
    with pytest.raises(ValueError, match="max_seed_results"):
        RelationshipExpander(reranker, [seed], max_seed_results=-1)
    with pytest.raises(RelationshipExpansionError, match="unique"):
        RelationshipExpander(reranker, [seed, seed])

    expander = RelationshipExpander(reranker, [seed])
    with pytest.raises(RelationshipExpansionError, match="missing from corpus"):
        expander.retrieve("query")
