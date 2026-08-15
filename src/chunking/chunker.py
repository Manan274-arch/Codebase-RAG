"""Split whole-file source documents into a shared code-chunk corpus."""

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

LANGUAGE_TO_LANGCHAIN = {
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


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split source documents while preserving their order and metadata."""
    _validate_chunk_settings(chunk_size, chunk_overlap)
    chunks: list[Document] = []

    for document in documents:
        splitter = _create_splitter(
            document.metadata.get("language"), chunk_size, chunk_overlap
        )
        document_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(document_chunks):
            start_index = chunk.metadata["start_index"]
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["start_line"] = _line_number_at(
                document.page_content, start_index
            )
            chunk.metadata["end_line"] = _line_number_at(
                document.page_content,
                start_index + len(chunk.page_content) - 1,
            )
            chunks.append(chunk)

    return chunks


def _create_splitter(
    language: object, chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    if isinstance(language, str) and language in LANGUAGE_TO_LANGCHAIN:
        return RecursiveCharacterTextSplitter.from_language(
            LANGUAGE_TO_LANGCHAIN[language],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=False,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=False,
    )


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _line_number_at(source: str, character_index: int) -> int:
    """Return the one-based line containing ``character_index``."""
    return source.count("\n", 0, character_index) + 1
