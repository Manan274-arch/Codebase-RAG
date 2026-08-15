from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from evaluation.metrics import (
    RetrievalEvaluationExample,
    canonical_chunk_id,
    evaluate_retriever,
)
from evaluation.rrf import RRFSearchResult
from langchain_core.documents import Document
from src.retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    RerankingError,
)


def document(name: str) -> Document:
    return Document(
        page_content=f"Source: {name}.py\n\nDefinitions:\n- {name}\n\nCode:\n{name}()",
        metadata={
            "source": f"{name}.py",
            "chunk_index": 0,
            "raw_content": f"{name}()",
            "marker": name,
        },
    )


class FakeCandidateRetriever:
    def __init__(self, candidates: Sequence[RRFSearchResult]) -> None:
        self.candidates = list(candidates)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 10) -> list[RRFSearchResult]:
        self.calls.append((query, k))
        return self.candidates[:k]


class FakeScorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls: list[list[tuple[str, str]]] = []

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> npt.NDArray[np.float32]:
        self.calls.append(list(pairs))
        return self.scores[: len(pairs)]


def candidates(*names: str) -> tuple[RRFSearchResult, ...]:
    return tuple(
        RRFSearchResult(document=document(name), score=1.0 / (61 + index))
        for index, name in enumerate(names)
    )


def test_fake_cross_encoder_reranks_rrf_candidates() -> None:
    first_stage = FakeCandidateRetriever(candidates("A", "B", "C"))
    scorer = FakeScorer([0.2, 0.9, 0.5])

    results = CrossEncoderReranker(first_stage, scorer=scorer).retrieve("query")

    assert [item.document.metadata["marker"] for item in results] == ["B", "C", "A"]
    assert [item.score for item in results] == pytest.approx([0.9, 0.5, 0.2])
    assert [item.first_stage_rank for item in results] == [2, 3, 1]
    assert results[0].first_stage_score == first_stage.candidates[1].score


def test_scorer_receives_enriched_page_content_not_raw_content() -> None:
    candidate = candidates("route")[0]
    scorer = FakeScorer([1.0])

    CrossEncoderReranker(
        FakeCandidateRetriever([candidate]), scorer=scorer
    ).retrieve("find route")

    assert scorer.calls == [[("find route", candidate.document.page_content)]]
    assert scorer.calls[0][0][1] != candidate.document.metadata["raw_content"]
    assert "Definitions:" in scorer.calls[0][0][1]


def test_candidate_depth_and_final_top_k_are_independent() -> None:
    first_stage = FakeCandidateRetriever(candidates("A", "B", "C"))
    scorer = FakeScorer([0.1, 0.8, 0.9])
    reranker = CrossEncoderReranker(
        first_stage, scorer=scorer, candidate_depth=3
    )

    results = reranker.retrieve("query", k=1)

    assert first_stage.calls == [("query", 3)]
    assert len(results) == 1
    assert results[0].document.metadata["marker"] == "C"


def test_ties_preserve_original_rrf_order_deterministically() -> None:
    first_stage = FakeCandidateRetriever(candidates("A", "B", "C"))
    reranker = CrossEncoderReranker(first_stage, scorer=FakeScorer([0.5, 0.5, 0.5]))

    runs = [reranker.retrieve("query") for _ in range(3)]

    assert all(
        [item.document.metadata["marker"] for item in run] == ["A", "B", "C"]
        for run in runs
    )


def test_original_documents_and_metadata_are_preserved_without_mutation() -> None:
    first_stage = FakeCandidateRetriever(candidates("A", "B"))
    originals = [item.document for item in first_stage.candidates]
    metadata_before = [dict(item.metadata) for item in originals]

    results = CrossEncoderReranker(
        first_stage, scorer=FakeScorer([0.1, 0.9])
    ).retrieve("query")

    assert results[0].document is originals[1]
    assert canonical_chunk_id(results[0].document) == "B.py::0"
    assert [item.metadata for item in originals] == metadata_before


def test_empty_one_candidate_blank_and_zero_behavior() -> None:
    empty_stage = FakeCandidateRetriever([])
    unused_scorer = FakeScorer([])
    empty = CrossEncoderReranker(empty_stage, scorer=unused_scorer)

    assert empty.retrieve("query") == []
    assert unused_scorer.calls == []

    one_stage = FakeCandidateRetriever(candidates("A"))
    one = CrossEncoderReranker(one_stage, scorer=FakeScorer([0.7]))
    assert one.retrieve("query", k=10)[0].document.metadata["marker"] == "A"
    calls_before = list(one_stage.calls)
    assert one.retrieve(" ") == []
    assert one.retrieve("query", k=0) == []
    assert one_stage.calls == calls_before


def test_invalid_arguments_and_scorer_outputs_fail_clearly() -> None:
    first_stage = FakeCandidateRetriever(candidates("A", "B"))
    with pytest.raises(ValueError, match="candidate_depth"):
        CrossEncoderReranker(first_stage, scorer=FakeScorer([]), candidate_depth=0)
    with pytest.raises(ValueError, match="model_name"):
        CrossEncoderReranker(first_stage, scorer=FakeScorer([]), model_name="")

    reranker = CrossEncoderReranker(first_stage, scorer=FakeScorer([0.1]))
    with pytest.raises(RerankingError, match="shape"):
        reranker.retrieve("query")
    with pytest.raises(ValueError, match="k"):
        reranker.retrieve("query", k=-1)

    non_finite = CrossEncoderReranker(
        first_stage, scorer=FakeScorer([float("nan"), 0.2])
    )
    with pytest.raises(RerankingError, match="non-finite"):
        non_finite.retrieve("query")


def test_generic_evaluator_accepts_reranked_results() -> None:
    first_stage = FakeCandidateRetriever(candidates("wrong", "right"))
    reranker = CrossEncoderReranker(first_stage, scorer=FakeScorer([0.1, 0.9]))
    corpus = [item.document for item in first_stage.candidates]
    examples = [
        RetrievalEvaluationExample(
            query_id="right",
            query="find right",
            relevant_chunk_ids=frozenset({"right.py::0"}),
        )
    ]

    evaluation = evaluate_retriever(reranker, corpus, examples, ks=(1, 3))

    assert evaluation.metrics[1].hit_rate == 1.0
    assert evaluation.queries[0].retrieved_chunk_ids[0] == "right.py::0"


def test_default_model_name_is_configurable_constant() -> None:
    assert DEFAULT_CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
