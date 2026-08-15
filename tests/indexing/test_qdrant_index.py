from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.documents import Document
from qdrant_client.http import models
from src.config import QdrantSettings
from src.indexing.qdrant_index import (
    QdrantCodeIndex,
    QdrantIndexError,
    StaleQdrantIndexError,
    document_from_payload,
    document_payload,
    point_id,
)
from src.retrieval.candidate_union import CandidateUnionRetriever
from src.retrieval.contracts import canonical_chunk_id
from src.retrieval.qdrant_dense import QdrantDenseRetriever


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: Sequence[str], *, normalize_embeddings: bool) -> np.ndarray:
        assert normalize_embeddings is True
        self.calls.append(tuple(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def document(name: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"{name}.py",
            "chunk_index": 0,
            "language": "python",
            "nested": [{"line": 1}],
        },
        id=f"doc-{name}",
    )


def settings(path: Path, *, vector_size: int = 2) -> QdrantSettings:
    return QdrantSettings(
        path=path,
        collection_name="test_chunks",
        embedding_model="fake-v1",
        vector_size=vector_size,
    )


def test_collection_creation_index_and_payload_round_trip(tmp_path: Path) -> None:
    documents = [document("a", "alpha"), document("b", "beta")]
    encoder = FakeEncoder({"alpha": [1, 0], "beta": [0, 1]})
    index = QdrantCodeIndex(settings(tmp_path / "qdrant"))
    result = index.ensure(documents, encoder)
    assert result.reused is False and result.point_count == 2
    assert index.client.collection_exists("test_chunks")
    assert index.client.count("test_chunks", exact=True).count == 2
    records, _ = index.client.scroll("test_chunks", limit=10, with_payload=True)
    reconstructed = [document_from_payload(record.payload) for record in records]
    assert {canonical_chunk_id(item) for item in reconstructed} == {
        "a.py::0",
        "b.py::0",
    }
    assert {item.id for item in reconstructed} == {"doc-a", "doc-b"}
    assert reconstructed[0].metadata["nested"] == [{"line": 1}]
    index.close()


def test_persistent_matching_index_is_reused_without_embedding(tmp_path: Path) -> None:
    documents = [document("a", "alpha")]
    path = tmp_path / "persistent"
    first_encoder = FakeEncoder({"alpha": [1, 0]})
    first = QdrantCodeIndex(settings(path))
    first.ensure(documents, first_encoder)
    first.close()
    unused_encoder = FakeEncoder({})
    reopened = QdrantCodeIndex(settings(path))
    result = reopened.ensure(documents, unused_encoder)
    assert result.reused is True
    assert unused_encoder.calls == []
    reopened.close()


def test_changed_corpus_requires_explicit_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "stale"
    original = [document("a", "alpha")]
    index = QdrantCodeIndex(settings(path))
    index.ensure(original, FakeEncoder({"alpha": [1, 0]}))
    changed = [document("a", "changed")]
    with pytest.raises(StaleQdrantIndexError, match="explicit rebuild"):
        index.ensure(changed, FakeEncoder({"changed": [0, 1]}))
    rebuilt = index.ensure(changed, FakeEncoder({"changed": [0, 1]}), rebuild=True)
    assert rebuilt.reused is False and rebuilt.point_count == 1
    index.close()


def test_dimension_validation_does_not_silently_build_bad_index(
    tmp_path: Path,
) -> None:
    index = QdrantCodeIndex(settings(tmp_path / "dimension", vector_size=3))
    with pytest.raises(QdrantIndexError, match="dimension 2"):
        index.ensure([document("a", "alpha")], FakeEncoder({"alpha": [1, 0]}))
    index.close()


def test_duplicate_canonical_identity_is_rejected_before_embedding(
    tmp_path: Path,
) -> None:
    item = document("a", "alpha")
    encoder = FakeEncoder({"alpha": [1, 0]})
    index = QdrantCodeIndex(settings(tmp_path / "duplicates"))
    with pytest.raises(QdrantIndexError, match="identities must be unique"):
        index.ensure([item, item], encoder)
    assert encoder.calls == []
    index.close()


def test_canonical_point_id_is_stable_and_payload_rejects_mismatch() -> None:
    item = document("a", "alpha")
    assert point_id("a.py::0") == point_id("a.py::0")
    assert point_id("a.py::0") != point_id("b.py::0")
    payload = document_payload(item)
    payload["canonical_id"] = "wrong.py::0"
    with pytest.raises(QdrantIndexError, match="inconsistent"):
        document_from_payload(payload)


def test_qdrant_dense_top_k_and_candidate_union_compatibility(
    tmp_path: Path,
) -> None:
    a, b = document("a", "alpha"), document("b", "beta")
    encoder = FakeEncoder({"alpha": [1, 0], "beta": [0, 1], "question": [0.9, 0.1]})
    index = QdrantCodeIndex(settings(tmp_path / "retrieve"))
    index.ensure([a, b], encoder)
    dense = QdrantDenseRetriever(index, encoder, search_mode="exact")
    results = dense.retrieve("question", k=2)
    assert [canonical_chunk_id(result.document) for result in results] == [
        "a.py::0",
        "b.py::0",
    ]
    assert all(
        result.document is not a and result.document is not b for result in results
    )
    union = CandidateUnionRetriever(dense, dense, bm25_depth=2, dense_depth=2)
    assert [result.provenance for result in union.retrieve("question", k=2)] == [
        "both",
        "both",
    ]
    assert encoder.calls.count(("alpha", "beta")) == 1
    index.close()


@pytest.mark.parametrize(
    ("mode", "expected_exact", "hnsw_ef"),
    [("exact", True, 64), ("hnsw", False, 256)],
)
def test_search_mode_and_hnsw_ef_reach_qdrant_query(
    tmp_path: Path, mode: str, expected_exact: bool, hnsw_ef: int
) -> None:
    index = QdrantCodeIndex(settings(tmp_path / mode))
    index.ensure([document("a", "alpha")], FakeEncoder({"alpha": [1, 0]}))
    response = models.QueryResponse(points=[])
    with patch.object(index.client, "query_points", return_value=response) as query:
        index.query(
            np.asarray([1, 0], dtype=np.float32),
            k=1,
            mode=mode,
            hnsw_ef=hnsw_ef,
        )
    params = query.call_args.kwargs["search_params"]
    assert params.exact is expected_exact
    assert params.hnsw_ef == hnsw_ef
    index.close()


def test_local_mode_explicitly_does_not_claim_real_hnsw(tmp_path: Path) -> None:
    index = QdrantCodeIndex(settings(tmp_path / "local"))
    assert index.supports_real_hnsw is False
    index.close()
