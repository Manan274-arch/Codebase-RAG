from pathlib import Path

import pytest
from langchain_core.documents import Document
from src.ingestion.loader import EXTENSION_TO_LANGUAGE, load_source_documents
from src.ingestion.repository import SUPPORTED_SOURCE_EXTENSIONS


def write_source(root: Path, relative_path: str, content: str = "") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_loads_one_document_with_exact_content_and_metadata(tmp_path: Path) -> None:
    content = "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"
    write_source(tmp_path, "backend/auth.py", content)

    documents = load_source_documents(tmp_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == content
    assert documents[0].metadata == {
        "source": "backend/auth.py",
        "language": "python",
    }


def test_loads_one_document_per_discovered_file(tmp_path: Path) -> None:
    write_source(tmp_path, "main.py", "print('hello')\n")
    write_source(tmp_path, "web/app.ts", "export const app = true;\n")
    write_source(tmp_path, "README.md", "Not source code")
    write_source(tmp_path, "node_modules/vendor.js", "ignored")

    documents = load_source_documents(tmp_path)

    assert len(documents) == 2
    assert [document.metadata["source"] for document in documents] == [
        "main.py",
        "web/app.ts",
    ]


@pytest.mark.parametrize(
    ("file_name", "expected_language"),
    [
        ("main.py", "python"),
        ("Main.java", "java"),
        ("main.c", "c"),
        ("main.h", "c"),
        ("main.cc", "cpp"),
        ("main.cpp", "cpp"),
        ("main.cxx", "cpp"),
        ("main.hh", "cpp"),
        ("main.hpp", "cpp"),
        ("main.hxx", "cpp"),
        ("app.js", "javascript"),
        ("app.jsx", "javascript"),
        ("app.mjs", "javascript"),
        ("app.cjs", "javascript"),
        ("app.ts", "typescript"),
        ("app.tsx", "typescript"),
        ("main.go", "go"),
        ("lib.rs", "rust"),
        ("Program.cs", "csharp"),
        ("app.rb", "ruby"),
        ("index.php", "php"),
        ("App.swift", "swift"),
        ("Main.kt", "kotlin"),
        ("build.kts", "kotlin"),
        ("Main.scala", "scala"),
        ("run.sh", "shell"),
        ("run.bash", "shell"),
        ("run.zsh", "shell"),
        ("query.sql", "sql"),
    ],
)
def test_classifies_supported_languages(
    tmp_path: Path, file_name: str, expected_language: str
) -> None:
    write_source(tmp_path, file_name)

    document = load_source_documents(tmp_path)[0]

    assert document.metadata["language"] == expected_language


def test_preserves_discovery_order_and_posix_relative_sources(tmp_path: Path) -> None:
    write_source(tmp_path, "z.py")
    write_source(tmp_path, "nested/b.ts")
    write_source(tmp_path, "nested/a.java")

    documents = load_source_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "nested/a.java",
        "nested/b.ts",
        "z.py",
    ]
    assert all("\\" not in document.metadata["source"] for document in documents)
    assert all(
        str(tmp_path) not in document.metadata["source"] for document in documents
    )


def test_classifies_extensions_case_insensitively(tmp_path: Path) -> None:
    write_source(tmp_path, "MODULE.PY")
    write_source(tmp_path, "Component.TsX")

    documents = load_source_documents(tmp_path)

    assert [document.metadata["language"] for document in documents] == [
        "typescript",
        "python",
    ]


def test_invalid_utf8_propagates_unicode_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.py"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(UnicodeDecodeError):
        load_source_documents(tmp_path)


def test_nonexistent_repository_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_source_documents(tmp_path / "missing")


def test_file_repository_path_raises_not_a_directory_error(tmp_path: Path) -> None:
    path = tmp_path / "main.py"
    path.touch()

    with pytest.raises(NotADirectoryError):
        load_source_documents(path)


def test_every_discovered_extension_has_a_language_mapping() -> None:
    assert SUPPORTED_SOURCE_EXTENSIONS == EXTENSION_TO_LANGUAGE.keys()
