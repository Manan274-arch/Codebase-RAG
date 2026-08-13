"""Load discovered source files as LangChain documents."""

from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.repository import find_source_files

EXTENSION_TO_LANGUAGE = {
    ".bash": "shell",
    ".c": "c",
    ".cc": "cpp",
    ".cjs": "javascript",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".cxx": "cpp",
    ".go": "go",
    ".h": "c",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".mjs": "javascript",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".zsh": "shell",
}


def load_source_documents(repo_path: Path) -> list[Document]:
    """Load each discovered source file into one LangChain ``Document``."""
    documents: list[Document] = []

    for relative_path in find_source_files(repo_path):
        documents.append(
            Document(
                page_content=(repo_path / relative_path).read_text(encoding="utf-8"),
                metadata={
                    "source": relative_path.as_posix(),
                    "language": EXTENSION_TO_LANGUAGE[
                        relative_path.suffix.casefold()
                    ],
                },
            )
        )

    return documents

