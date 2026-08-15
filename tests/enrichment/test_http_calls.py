from pathlib import Path

import pytest
from langchain_core.documents import Document
from src.chunking.chunker import chunk_documents
from src.enrichment.metadata import enrich_chunks
from src.enrichment.structure import extract_structure
from src.ingestion.loader import load_source_documents


def document(source: str, language: str, path: str = "source") -> Document:
    return Document(
        page_content=source,
        metadata={"source": path, "language": language},
    )


@pytest.mark.parametrize("language", ["javascript", "typescript"])
def test_fetch_default_get_and_explicit_post(language: str) -> None:
    source = (
        'fetch("/api/users");\n'
        'fetch("/api/users", { method: "POST" });\n'
    )

    calls = extract_structure(document(source, language)).http_calls

    assert [
        (call.client, call.method, call.target, call.caller) for call in calls
    ] == [
        ("fetch", "GET", "/api/users", None),
        ("fetch", "POST", "/api/users", None),
    ]
    assert [(call.span.start_line, call.span.end_line) for call in calls] == [
        (1, 1),
        (2, 2),
    ]


@pytest.mark.parametrize("language", ["javascript", "typescript"])
def test_axios_methods_interpolation_and_lexical_caller(language: str) -> None:
    source = (
        "async function getUser(id) {\n"
        "    return axios.get(`/api/users/${id}`);\n"
        "}\n"
        'axios.post("/api/users", data);\n'
        'axios.put("/api/users/1", data);\n'
        'axios.patch("/api/users/1", data);\n'
        'axios.delete("/api/users/1");\n'
        'axios.head("/api/users/1");\n'
        'axios.options("/api/users/1");\n'
    )

    calls = extract_structure(document(source, language)).http_calls

    assert (calls[0].client, calls[0].method, calls[0].target, calls[0].caller) == (
        "axios",
        "GET",
        "/api/users/{id}",
        "getUser",
    )
    assert [call.method for call in calls[1:]] == [
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "HEAD",
        "OPTIONS",
    ]


def test_python_requests_httpx_fstring_and_caller() -> None:
    source = (
        "import requests\n"
        "import httpx\n"
        "def load_user(user_id):\n"
        '    requests.get(f"/api/users/{user_id}")\n'
        '    requests.post("/api/users", json={})\n'
        '    httpx.get("https://example.com/users")\n'
    )

    calls = extract_structure(document(source, "python")).http_calls

    assert [
        (call.client, call.method, call.target, call.caller) for call in calls
    ] == [
        ("requests", "GET", "/api/users/{user_id}", "load_user"),
        ("requests", "POST", "/api/users", "load_user"),
        ("httpx", "GET", "https://example.com/users", "load_user"),
    ]


def test_python_supported_method_matrix() -> None:
    source = "\n".join(
        f'requests.{method}("/target")'
        for method in ("get", "post", "put", "patch", "delete", "head", "options")
    )

    calls = extract_structure(document(source, "python")).http_calls

    assert [call.method for call in calls] == [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "HEAD",
        "OPTIONS",
    ]


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "javascript",
            '// fetch("/fake");\nconst text = \'axios.get("/fake")\';\n',
        ),
        (
            "python",
            '# requests.get("/fake")\ntext = \'httpx.post("/fake")\'\n',
        ),
    ],
)
def test_comments_and_strings_do_not_create_http_calls(
    language: str, source: str
) -> None:
    assert extract_structure(document(source, language)).http_calls == ()


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", 'send_request("/api/users")\n'),
        ("javascript", 'superagent.get("/api/users");\n'),
        (
            "javascript",
            'fetch(API_BASE + "/users");\naxios.get(makeUrl());\n',
        ),
        (
            "java",
            'class Api { void call() { client.send("/api/users"); } }\n',
        ),
    ],
)
def test_unsupported_clients_and_dynamic_targets_yield_no_calls(
    language: str, source: str
) -> None:
    assert extract_structure(document(source, language)).http_calls == ()


def test_http_call_reaches_the_correct_chunk_end_to_end(tmp_path: Path) -> None:
    source = (
        'import axios from "axios";\n\n'
        "export async function getUser(id) {\n"
        "    return axios.get(`/api/users/${id}`);\n"
        "}\n"
    )
    (tmp_path / "api.ts").write_text(source, encoding="utf-8")

    whole_file = load_source_documents(tmp_path)[0]
    chunks = chunk_documents([whole_file], chunk_size=55, chunk_overlap=5)
    enriched = enrich_chunks(chunks, extract_structure(whole_file))
    matching = [chunk for chunk in enriched if chunk.metadata["structural_http_calls"]]

    assert matching
    assert any("axios.get" in chunk.page_content for chunk in matching)
    assert matching[0].metadata["structural_http_calls"][0] == {
        "method": "GET",
        "target": "/api/users/{id}",
        "client": "axios",
        "caller": "getUser",
        "start_line": 4,
        "end_line": 4,
    }
