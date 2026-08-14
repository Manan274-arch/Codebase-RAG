import pytest
from langchain_core.documents import Document
from src.retrieval.bm25 import BM25Retriever, tokenize


def document(content: str, **metadata: object) -> Document:
    return Document(page_content=content, metadata=metadata)


def test_distinctive_page_content_ranks_above_irrelevant_documents() -> None:
    documents = [
        document("parse configuration values", source="config.py"),
        document("validate authentication_token", source="auth.py"),
        document("render user interface", source="view.ts"),
    ]

    results = BM25Retriever(documents).retrieve("authentication_token")

    assert results[0].document is documents[1]
    assert results[0].score > results[1].score


def test_metadata_is_excluded_but_page_content_is_indexed() -> None:
    metadata_only = document(
        "completely unrelated implementation",
        source="metadata.py",
        structural_definitions=[{"name": "UniqueMetadataOnlyToken"}],
        structural_routes=[{"path": "/UniqueMetadataOnlyToken"}],
        related_route_chunks=[{"source": "UniqueMetadataOnlyToken"}],
    )
    content_match = document(
        "def UniqueMetadataOnlyToken(): pass",
        source="content.py",
    )
    irrelevant = document("another unrelated implementation", source="other.py")

    metadata_results = BM25Retriever([metadata_only, irrelevant]).retrieve(
        "UniqueMetadataOnlyToken"
    )
    content_results = BM25Retriever(
        [metadata_only, content_match, irrelevant]
    ).retrieve("UniqueMetadataOnlyToken")

    assert [result.score for result in metadata_results] == [0.0, 0.0]
    assert metadata_results[0].document is metadata_only
    assert content_results[0].document is content_match
    assert content_results[0].score > content_results[1].score


def test_top_k_is_respected_and_larger_than_corpus_is_safe() -> None:
    documents = [document("alpha"), document("beta"), document("gamma")]
    retriever = BM25Retriever(documents)

    assert len(retriever.retrieve("alpha", k=2)) == 2
    assert len(retriever.retrieve("alpha", k=20)) == 3
    assert retriever.retrieve("alpha", k=0) == []


def test_tied_scores_preserve_corpus_order() -> None:
    documents = [
        document("first content", position=0),
        document("second content", position=1),
        document("third content", position=2),
    ]

    results = BM25Retriever(documents).retrieve("absent_token")

    assert [result.document for result in results] == documents
    assert [result.score for result in results] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("query", ["", "   \t\n", "---...!!!"])
def test_empty_or_tokenless_query_returns_no_results(query: str) -> None:
    assert BM25Retriever([document("content")]).retrieve(query) == []


def test_empty_corpus_is_safe() -> None:
    retriever = BM25Retriever([])

    assert retriever.retrieve("anything") == []
    assert retriever.retrieve("") == []


def test_corpus_with_only_tokenless_content_returns_stable_zero_scores() -> None:
    documents = [document("---"), document("   ")]

    results = BM25Retriever(documents).retrieve("anything")

    assert [result.document for result in results] == documents
    assert [result.score for result in results] == [0.0, 0.0]


def test_original_documents_content_metadata_and_identity_remain_intact() -> None:
    original = document(
        "lookup exact_identifier",
        source="service.py",
        chunk_index=3,
        structural_routes=[{"path": "/users"}],
    )
    original_metadata = dict(original.metadata)

    result = BM25Retriever([original]).retrieve("exact_identifier")[0]

    assert result.document is original
    assert result.document.page_content == "lookup exact_identifier"
    assert result.document.metadata == original_metadata
    assert original.metadata == original_metadata


def test_tokenizer_lowercases_and_preserves_identifier_underscores() -> None:
    assert tokenize("HTTPServer user_id value-2") == [
        "httpserver",
        "user_id",
        "value",
        "2",
    ]


def test_negative_k_fails_clearly() -> None:
    with pytest.raises(ValueError, match="k must be non-negative"):
        BM25Retriever([document("content")]).retrieve("content", k=-1)
