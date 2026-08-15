from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from src.chunking.chunker import chunk_documents
from src.enrichment.metadata import enrich_chunks
from src.enrichment.relationships import (
    RelationshipLinkingError,
    link_http_calls_to_routes,
)
from src.enrichment.structure import extract_structure
from src.ingestion.loader import load_source_documents


def chunk(
    source: str,
    chunk_index: int,
    *,
    routes: list[dict[str, Any]] | None = None,
    calls: list[dict[str, Any]] | None = None,
    content: str = "original code",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "language": "python",
            "chunk_index": chunk_index,
            "start_index": 0,
            "start_line": chunk_index * 10 + 1,
            "end_line": chunk_index * 10 + 10,
            "structural_definitions": [{"name": "preserved"}],
            "structural_imports": [{"source": "preserved"}],
            "structural_routes": routes or [],
            "structural_http_calls": calls or [],
        },
        id=f"existing-{source}-{chunk_index}",
    )


def route(path: str, methods: list[str]) -> dict[str, Any]:
    return {
        "path": path,
        "methods": methods,
        "framework": "test",
        "handler": "handler",
        "start_line": 1,
        "end_line": 2,
    }


def call(target: str, method: str | None) -> dict[str, Any]:
    return {
        "target": target,
        "method": method,
        "client": "test",
        "caller": "caller",
        "start_line": 1,
        "end_line": 1,
    }


def reference(source: str, chunk_index: int) -> dict[str, object]:
    return {
        "source": source,
        "chunk_index": chunk_index,
        "start_line": chunk_index * 10 + 1,
        "end_line": chunk_index * 10 + 10,
    }


def test_exact_match_creates_bidirectional_references() -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/api/users", "GET")])
    backend = chunk("backend.py", 0, routes=[route("/api/users", ["GET"])])

    result = link_http_calls_to_routes([caller, backend])

    assert result[0].metadata["related_route_chunks"] == [
        reference("backend.py", 0)
    ]
    assert result[1].metadata["related_http_call_chunks"] == [
        reference("frontend.ts", 0)
    ]


@pytest.mark.parametrize("backend_path", ["/api/users/{id}", "/api/users/:id"])
def test_parameterized_backend_route_matches_one_call_segment(
    backend_path: str,
) -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/api/users/123", "GET")])
    backend = chunk("backend.py", 0, routes=[route(backend_path, ["GET"])])

    result = link_http_calls_to_routes([caller, backend])

    assert result[0].metadata["related_route_chunks"]


def test_interpolated_parameter_names_need_not_match() -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/api/users/{id}", "GET")])
    backend = chunk(
        "backend.py", 0, routes=[route("/api/users/{user_id}", ["GET"])]
    )

    result = link_http_calls_to_routes([caller, backend])

    assert result[0].metadata["related_route_chunks"]


def test_method_mismatch_does_not_match_but_unrestricted_spring_route_does() -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/api/users", "GET")])
    post = chunk("post.py", 0, routes=[route("/api/users", ["POST"])])
    unrestricted = chunk("spring.java", 0, routes=[route("/api/users", [])])

    result = link_http_calls_to_routes([caller, post, unrestricted])

    assert result[0].metadata["related_route_chunks"] == [
        reference("spring.java", 0)
    ]
    assert result[1].metadata["related_http_call_chunks"] == []


def test_literal_route_wins_over_parameterized_route() -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/users/me", "GET")])
    parameterized = chunk(
        "parameter.py", 0, routes=[route("/users/{id}", ["GET"])]
    )
    literal = chunk("literal.py", 0, routes=[route("/users/me", ["GET"])])

    result = link_http_calls_to_routes([caller, parameterized, literal])

    assert result[0].metadata["related_route_chunks"] == [
        reference("literal.py", 0)
    ]
    assert result[1].metadata["related_http_call_chunks"] == []


def test_equal_specificity_matches_and_overlapping_route_chunks_are_retained() -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/users/123", "GET")])
    first = chunk("a.py", 0, routes=[route("/users/{id}", ["GET"])])
    second = chunk("b.py", 0, routes=[route("/users/{user_id}", ["GET"])])
    overlap = chunk("b.py", 1, routes=[route("/users/{user_id}", ["GET"])])

    result = link_http_calls_to_routes([caller, overlap, second, first])

    assert result[0].metadata["related_route_chunks"] == [
        reference("a.py", 0),
        reference("b.py", 0),
        reference("b.py", 1),
    ]


@pytest.mark.parametrize(
    "target",
    ["/api/users?page=2", "/api/users#section", "/api/users/"],
)
def test_comparison_normalizes_query_fragment_and_trailing_slash(target: str) -> None:
    caller = chunk("frontend.ts", 0, calls=[call(target, "GET")])
    backend = chunk("backend.py", 0, routes=[route("/api/users", ["GET"])])

    result = link_http_calls_to_routes([caller, backend])

    assert result[0].metadata["related_route_chunks"]
    assert result[0].metadata["structural_http_calls"][0]["target"] == target


@pytest.mark.parametrize(
    "outbound_call",
    [
        call("https://example.com/api/users", "GET"),
        call("//example.com/api/users", "GET"),
        call("/api/users", None),
        call("api/users", "GET"),
    ],
)
def test_ineligible_external_relative_or_unknown_method_calls_do_not_match(
    outbound_call: dict[str, Any],
) -> None:
    caller = chunk("frontend.ts", 0, calls=[outbound_call])
    backend = chunk("backend.py", 0, routes=[route("/api/users", ["GET"])])

    result = link_http_calls_to_routes([caller, backend])

    assert result[0].metadata["related_route_chunks"] == []
    assert result[1].metadata["related_http_call_chunks"] == []


def test_call_side_placeholder_does_not_override_literal_backend_authority() -> None:
    caller = chunk("frontend.ts", 0, calls=[call("/users/{id}", "GET")])
    backend = chunk("backend.py", 0, routes=[route("/users/admin", ["GET"])])

    result = link_http_calls_to_routes([caller, backend])

    assert result[0].metadata["related_route_chunks"] == []


def test_no_match_preserves_documents_metadata_content_ids_and_order() -> None:
    chunks = [
        chunk("frontend.ts", 0, calls=[call("/missing", "GET")], content="call"),
        chunk("backend.py", 0, routes=[route("/other", ["GET"])], content="route"),
    ]
    original_metadata = [dict(item.metadata) for item in chunks]

    result = link_http_calls_to_routes(chunks)

    assert result is not chunks
    assert [item.page_content for item in result] == ["call", "route"]
    assert [item.id for item in result] == [item.id for item in chunks]
    assert [item.metadata["source"] for item in result] == [
        "frontend.ts",
        "backend.py",
    ]
    assert [item.metadata for item in chunks] == original_metadata
    assert all(item.metadata["related_route_chunks"] == [] for item in result)
    assert all(item.metadata["related_http_call_chunks"] == [] for item in result)
    for before, after in zip(chunks, result, strict=True):
        for key in (
            "language",
            "chunk_index",
            "start_line",
            "end_line",
            "structural_definitions",
            "structural_imports",
            "structural_routes",
            "structural_http_calls",
        ):
            assert after.metadata[key] == before.metadata[key]


def test_duplicate_chunk_identity_fails_clearly() -> None:
    with pytest.raises(RelationshipLinkingError, match="uniquely identify"):
        link_http_calls_to_routes([chunk("same.py", 0), chunk("same.py", 0)])


def test_cross_language_repository_pipeline_links_original_chunks(
    tmp_path: Path,
) -> None:
    backend_source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        '@app.get("/api/users/{user_id}")\n'
        "def get_user(user_id: str):\n"
        "    return {\"id\": user_id}\n"
    )
    frontend_source = (
        'import axios from "axios";\n'
        "export async function getUser(id: string) {\n"
        "    return axios.get(`/api/users/${id}`);\n"
        "}\n"
    )
    (tmp_path / "backend.py").write_text(backend_source, encoding="utf-8")
    (tmp_path / "frontend.ts").write_text(frontend_source, encoding="utf-8")

    documents = load_source_documents(tmp_path)
    raw_chunks = chunk_documents(documents, chunk_size=65, chunk_overlap=5)
    enriched: list[Document] = []
    for document_item in documents:
        source = document_item.metadata["source"]
        source_chunks = [
            item for item in raw_chunks if item.metadata["source"] == source
        ]
        enriched.extend(enrich_chunks(source_chunks, extract_structure(document_item)))

    result = link_http_calls_to_routes(enriched)
    frontend = [
        item
        for item in result
        if item.metadata["source"] == "frontend.ts"
        and item.metadata["structural_http_calls"]
    ]
    backend = [
        item
        for item in result
        if item.metadata["source"] == "backend.py"
        and item.metadata["structural_routes"]
    ]

    assert frontend and backend
    assert frontend[0].metadata["related_route_chunks"]
    assert backend[0].metadata["related_http_call_chunks"]
    assert frontend[0].metadata["related_route_chunks"][0]["source"] == "backend.py"
    assert backend[0].metadata["related_http_call_chunks"][0]["source"] == "frontend.ts"
    assert [item.page_content for item in result] == [
        item.page_content for item in enriched
    ]
