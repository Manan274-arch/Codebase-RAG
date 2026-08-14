from pathlib import Path

from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents
from src.ingestion.enrichment import (
    StructuralEnrichmentError,
    enrich_chunks,
    line_ranges_overlap,
)
from src.ingestion.loader import load_source_documents
from src.ingestion.structure import (
    Definition,
    FileStructure,
    Import,
    RouteDefinition,
    SourceSpan,
    extract_structure,
)


def span(start_line: int, end_line: int) -> SourceSpan:
    return SourceSpan(start_line, end_line, 0, 1)


def definition(
    name: str,
    start_line: int,
    end_line: int,
    *,
    kind: str = "function",
    qualified_name: str | None = None,
) -> Definition:
    return Definition(
        name=name,
        kind=kind,
        span=span(start_line, end_line),
        qualified_name=qualified_name or name,
    )


def chunk(start_line: int, end_line: int, *, source: str = "app.py") -> Document:
    return Document(
        page_content="original code",
        metadata={
            "source": source,
            "language": "python",
            "chunk_index": 0,
            "start_index": 0,
            "start_line": start_line,
            "end_line": end_line,
            "owner": "team-a",
        },
        id="chunk-id",
    )


def structure(
    *,
    definitions: tuple[Definition, ...] = (),
    imports: tuple[Import, ...] = (),
    source: str = "app.py",
) -> FileStructure:
    return FileStructure("python", definitions, imports, source)


def test_inclusive_line_range_overlap_cases() -> None:
    assert line_ranges_overlap(10, 20, 12, 16)
    assert line_ranges_overlap(12, 16, 10, 20)
    assert line_ranges_overlap(10, 20, 20, 30)
    assert line_ranges_overlap(10, 20, 15, 30)
    assert not line_ranges_overlap(10, 20, 21, 30)


def test_definition_inside_chunk_is_attached() -> None:
    result = enrich_chunks(
        [chunk(12, 16)], structure(definitions=(definition("foo", 10, 20),))
    )

    assert [item["name"] for item in result[0].metadata["structural_definitions"]] == [
        "foo"
    ]


def test_definition_spanning_multiple_chunks_is_attached_to_each() -> None:
    chunks = [chunk(10, 18), chunk(19, 30)]

    result = enrich_chunks(
        chunks, structure(definitions=(definition("foo", 10, 30),))
    )

    assert all(
        item.metadata["structural_definitions"][0]["name"] == "foo"
        for item in result
    )


def test_nested_definitions_and_boundary_overlap_are_both_preserved() -> None:
    definitions = (
        definition(
            "get_user",
            20,
            35,
            kind="method",
            qualified_name="UserService.get_user",
        ),
        definition("UserService", 10, 80, kind="class"),
        definition("unrelated", 40, 50),
        definition("boundary", 1, 22),
    )

    result = enrich_chunks([chunk(22, 30)], structure(definitions=definitions))

    assert [
        item["qualified_name"]
        for item in result[0].metadata["structural_definitions"]
    ] == ["boundary", "UserService", "UserService.get_user"]


def test_imports_only_attach_to_overlapping_chunks() -> None:
    imported = Import("import os", (), None, False, span(3, 3))

    result = enrich_chunks(
        [chunk(1, 5), chunk(6, 10)], structure(imports=(imported,))
    )

    assert result[0].metadata["structural_imports"][0]["source"] == "import os"
    assert result[1].metadata["structural_imports"] == []


def test_enrichment_preserves_content_metadata_order_and_id_without_mutation() -> None:
    chunks = [chunk(1, 2), chunk(3, 4)]
    original_metadata = [dict(item.metadata) for item in chunks]

    result = enrich_chunks(chunks, structure())

    assert result is not chunks
    assert [item.page_content for item in result] == ["original code", "original code"]
    assert [item.id for item in result] == ["chunk-id", "chunk-id"]
    assert [item.metadata["chunk_index"] for item in result] == [0, 0]
    assert [item.metadata for item in chunks] == original_metadata
    assert all(item.metadata["structural_definitions"] == [] for item in result)
    assert all(item.metadata["structural_imports"] == [] for item in result)
    assert all(item.metadata["structural_routes"] == [] for item in result)


def test_routes_use_the_same_line_overlap_and_plain_metadata() -> None:
    route = RouteDefinition(
        path="/users/{id}",
        methods=("GET",),
        framework="fastapi",
        handler="get_user",
        span=span(10, 20),
        owner="app",
    )
    file_structure = FileStructure(
        language="python",
        definitions=(),
        imports=(),
        source="app.py",
        routes=(route,),
    )

    result = enrich_chunks([chunk(20, 25), chunk(26, 30)], file_structure)

    assert result[0].metadata["structural_routes"] == [
        {
            "path": "/users/{id}",
            "methods": ["GET"],
            "framework": "fastapi",
            "handler": "get_user",
            "owner": "app",
            "start_line": 10,
            "end_line": 20,
        }
    ]
    assert result[1].metadata["structural_routes"] == []


def test_duplicate_records_are_removed_and_output_is_source_ordered() -> None:
    early = definition("early", 2, 3)
    late = definition("late", 8, 9)

    result = enrich_chunks(
        [chunk(1, 10)], structure(definitions=(late, early, early))
    )

    assert [item["name"] for item in result[0].metadata["structural_definitions"]] == [
        "early",
        "late",
    ]


def test_missing_or_invalid_chunk_ranges_fail_clearly() -> None:
    missing = chunk(1, 2)
    del missing.metadata["start_line"]

    try:
        enrich_chunks([missing], structure())
    except StructuralEnrichmentError as error:
        assert "start_line" in str(error)
        assert "app.py" in str(error)
    else:
        raise AssertionError("missing start_line did not fail")

    try:
        enrich_chunks([chunk(4, 3)], structure())
    except StructuralEnrichmentError as error:
        assert "invalid chunk line range 4-3" in str(error)
    else:
        raise AssertionError("invalid range did not fail")


def test_source_mismatch_fails_clearly() -> None:
    try:
        enrich_chunks([chunk(1, 2, source="auth.py")], structure(source="users.py"))
    except StructuralEnrichmentError as error:
        assert "auth.py" in str(error)
        assert "users.py" in str(error)
    else:
        raise AssertionError("source mismatch did not fail")


def test_real_python_ingestion_flow_connects_chunks_and_structure(
    tmp_path: Path,
) -> None:
    source = (
        "import os\n\n"
        "class Service:\n"
        "    def method(self):\n"
        "        return os.getcwd()\n"
    )
    (tmp_path / "service.py").write_text(source, encoding="utf-8")
    document = load_source_documents(tmp_path)[0]

    chunks = chunk_documents([document], chunk_size=45, chunk_overlap=5)
    result = enrich_chunks(chunks, extract_structure(document))

    method_chunks = [
        item
        for item in result
        if any(
            definition["qualified_name"] == "Service.method"
            for definition in item.metadata["structural_definitions"]
        )
    ]
    assert method_chunks
    assert any("def method" in item.page_content for item in method_chunks)
    assert result[0].metadata["structural_imports"][0]["source"] == "import os"
    assert all(
        item.metadata["structural_imports"] == [] for item in result[1:]
    )
