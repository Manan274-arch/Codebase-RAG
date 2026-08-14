from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from langchain_core.documents import Document
from src.retrieval.dense import DenseRetrievalError, DenseRetriever
from src.retrieval.evaluation import RetrievalEvaluationExample, evaluate_retriever


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> npt.NDArray[np.float32]:
        self.calls.append((list(texts), normalize_embeddings))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def document(content: str, source: str, chunk_index: int) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "chunk_index": chunk_index,
            "structural_definitions": [{"name": "metadata_only_secret"}],
        },
    )


def test_corpus_embedding_uses_only_page_content_and_runs_once() -> None:
    documents = [
        document("first raw code", "a.py", 0),
        document("second raw code", "b.py", 0),
    ]
    encoder = FakeEncoder(
        {
            "first raw code": [1.0, 0.0],
            "second raw code": [0.0, 1.0],
            "query": [1.0, 0.0],
        }
    )

    retriever = DenseRetriever(documents, encoder=encoder)
    retriever.retrieve("query")
    retriever.retrieve("query")

    assert encoder.calls[0] == (
        ["first raw code", "second raw code"],
        True,
    )
    assert "metadata_only_secret" not in encoder.calls[0][0]
    assert encoder.calls.count((['first raw code', 'second raw code'], True)) == 1
    assert encoder.calls[1:] == [(["query"], True), (["query"], True)]


def test_cosine_ranking_scores_k_and_original_document_identity() -> None:
    documents = [
        document("horizontal", "a.py", 0),
        document("vertical", "b.py", 0),
        document("diagonal", "c.py", 0),
    ]
    encoder = FakeEncoder(
        {
            "horizontal": [1.0, 0.0],
            "vertical": [0.0, 1.0],
            "diagonal": [1.0, 1.0],
            "find vertical": [0.0, 2.0],
        }
    )
    original_metadata = [dict(item.metadata) for item in documents]

    results = DenseRetriever(documents, encoder=encoder).retrieve(
        "find vertical", k=2
    )

    assert [result.document for result in results] == [documents[1], documents[2]]
    assert results[0].document is documents[1]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1 / np.sqrt(2))
    assert [item.metadata for item in documents] == original_metadata


def test_similarity_ties_preserve_corpus_order() -> None:
    documents = [
        document("first", "a.py", 0),
        document("second", "b.py", 0),
    ]
    encoder = FakeEncoder(
        {"first": [1.0, 0.0], "second": [1.0, 0.0], "query": [1.0, 0.0]}
    )

    results = DenseRetriever(documents, encoder=encoder).retrieve("query")

    assert [result.document for result in results] == documents


def test_k_and_empty_input_behavior_matches_retrieval_conventions() -> None:
    document_item = document("code", "a.py", 0)
    encoder = FakeEncoder({"code": [1.0], "query": [1.0]})
    retriever = DenseRetriever([document_item], encoder=encoder)

    assert retriever.retrieve("query", k=0) == []
    assert len(retriever.retrieve("query", k=20)) == 1
    assert retriever.retrieve("   ") == []
    with pytest.raises(ValueError, match="k must be non-negative"):
        retriever.retrieve("query", k=-1)


def test_empty_corpus_does_not_invoke_encoder() -> None:
    encoder = FakeEncoder({"query": [1.0]})

    retriever = DenseRetriever([], encoder=encoder)

    assert retriever.retrieve("query") == []
    assert encoder.calls == []


def test_invalid_or_zero_embeddings_fail_clearly() -> None:
    zero_encoder = FakeEncoder({"code": [0.0, 0.0]})

    with pytest.raises(DenseRetrievalError, match="zero-length"):
        DenseRetriever([document("code", "a.py", 0)], encoder=zero_encoder)


def test_dense_results_work_with_retriever_agnostic_evaluator() -> None:
    documents = [
        document("authentication implementation", "auth.py", 0),
        document("render interface", "view.ts", 0),
    ]
    encoder = FakeEncoder(
        {
            "authentication implementation": [1.0, 0.0],
            "render interface": [0.0, 1.0],
            "where are credentials checked": [2.0, 0.0],
        }
    )
    retriever = DenseRetriever(documents, encoder=encoder)
    examples = [
        RetrievalEvaluationExample(
            query_id="auth",
            query="where are credentials checked",
            relevant_chunk_ids=frozenset({"auth.py::0"}),
        )
    ]

    result = evaluate_retriever(retriever, documents, examples, ks=(1,))

    assert result.metrics[1].hit_rate == 1.0
    assert result.queries[0].retrieved_chunk_ids[0] == "auth.py::0"
