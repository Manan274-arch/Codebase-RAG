from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import Language
from src.chunking.chunker import LANGUAGE_TO_LANGCHAIN, chunk_documents


def source_document(
    content: str,
    *,
    source: str = "src/main.py",
    language: str = "python",
    **metadata: Any,
) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "language": language, **metadata},
    )


def test_small_document_remains_one_chunk_with_metadata() -> None:
    document = source_document("def greet():\n    return 'hello'\n", owner="team-a")

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content
    assert chunks[0].metadata == {
        "source": "src/main.py",
        "language": "python",
        "owner": "team-a",
        "start_index": 0,
        "chunk_index": 0,
        "start_line": 1,
        "end_line": 2,
    }


def test_large_document_becomes_ordered_bounded_chunks() -> None:
    content = "\n".join(f"value_{index} = {index}" for index in range(100))

    chunks = chunk_documents(
        [source_document(content)], chunk_size=80, chunk_overlap=10
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 80 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert [chunk.metadata["start_index"] for chunk in chunks] == sorted(
        chunk.metadata["start_index"] for chunk in chunks
    )


def test_chunk_indices_restart_and_parent_order_is_preserved() -> None:
    first = source_document("a " * 100, source="a.py")
    second = source_document("b " * 100, source="b.py")

    chunks = chunk_documents([first, second], chunk_size=40, chunk_overlap=5)
    first_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "a.py"]
    second_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "b.py"]

    assert chunks == first_chunks + second_chunks
    assert [chunk.metadata["chunk_index"] for chunk in first_chunks] == list(
        range(len(first_chunks))
    )
    assert [chunk.metadata["chunk_index"] for chunk in second_chunks] == list(
        range(len(second_chunks))
    )


def test_start_indices_reference_original_content_and_overlap_is_applied() -> None:
    content = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_documents(
        [source_document(content, language="shell")],
        chunk_size=10,
        chunk_overlap=3,
    )

    for chunk in chunks:
        start_index = chunk.metadata["start_index"]
        assert content[start_index : start_index + len(chunk.page_content)] == (
            chunk.page_content
        )
    assert chunks[1].metadata["start_index"] < len(chunks[0].page_content)


def test_chunk_line_ranges_are_one_based_and_inclusive() -> None:
    content = "first line\nsecond line\nthird line\n"

    chunks = chunk_documents(
        [source_document(content, language="shell")],
        chunk_size=12,
        chunk_overlap=0,
    )

    for chunk in chunks:
        start = chunk.metadata["start_index"]
        end = start + len(chunk.page_content) - 1
        assert chunk.metadata["start_line"] == content.count("\n", 0, start) + 1
        assert chunk.metadata["end_line"] == content.count("\n", 0, end) + 1


EXPECTED_LANGUAGE_MAPPING = {
    "c": Language.C,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
    "go": Language.GO,
    "java": Language.JAVA,
    "javascript": Language.JS,
    "kotlin": Language.KOTLIN,
    "php": Language.PHP,
    "python": Language.PYTHON,
    "ruby": Language.RUBY,
    "rust": Language.RUST,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "typescript": Language.TS,
}


def test_normalized_languages_map_to_current_langchain_enum() -> None:
    assert LANGUAGE_TO_LANGCHAIN == EXPECTED_LANGUAGE_MAPPING


@pytest.mark.parametrize("language", EXPECTED_LANGUAGE_MAPPING)
def test_language_aware_languages_produce_chunks(language: str) -> None:
    document = source_document("function body\n" * 30, language=language)

    chunks = chunk_documents([document], chunk_size=60, chunk_overlap=5)

    assert chunks
    assert all(chunk.metadata["language"] == language for chunk in chunks)


@pytest.mark.parametrize("language", ["shell", "sql"])
def test_languages_without_langchain_support_use_fallback(language: str) -> None:
    document = source_document("statement value\n" * 30, language=language)

    chunks = chunk_documents([document], chunk_size=60, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(chunk.metadata["language"] == language for chunk in chunks)


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_documents([source_document("")]) == []


def test_mixed_language_documents_preserve_input_order() -> None:
    documents = [
        source_document("print('hello')", source="a.py", language="python"),
        source_document("SELECT 1;", source="b.sql", language="sql"),
        source_document("fn main() {}", source="c.rs", language="rust"),
    ]

    chunks = chunk_documents(documents)

    assert [chunk.metadata["source"] for chunk in chunks] == [
        "a.py",
        "b.sql",
        "c.rs",
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 0, 0]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be greater than zero"),
        (-1, 0, "chunk_size must be greater than zero"),
        (10, -1, "chunk_overlap must be non-negative"),
        (10, 10, "chunk_overlap must be smaller than chunk_size"),
        (10, 11, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_invalid_chunk_settings_fail_clearly(
    chunk_size: int, chunk_overlap: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_documents([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
