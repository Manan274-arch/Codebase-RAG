from pathlib import Path

import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents
from src.ingestion.enrichment import enrich_chunks
from src.ingestion.loader import load_source_documents
from src.ingestion.structure import extract_structure


def document(source: str, language: str, path: str = "source") -> Document:
    return Document(
        page_content=source,
        metadata={"source": path, "language": language},
    )


def test_fastapi_decorators_have_methods_handlers_spans_and_stable_order() -> None:
    source = (
        "from fastapi import FastAPI\n"
        "\n"
        "@ordinary_decorator\n"
        "@app.get(\"/users\")\n"
        "def list_users():\n"
        "    pass\n"
        "\n"
        "@router.post(\"/users\")\n"
        "def create_user():\n"
        "    pass\n"
    )

    routes = extract_structure(document(source, "python", "api.py")).routes

    actual = [
        (route.path, route.methods, route.framework, route.handler)
        for route in routes
    ]
    assert actual == [
        ("/users", ("GET",), "fastapi", "list_users"),
        ("/users", ("POST",), "fastapi", "create_user"),
    ]
    assert [(route.span.start_line, route.span.end_line) for route in routes] == [
        (3, 6),
        (8, 10),
    ]


def test_flask_route_default_explicit_methods_and_blueprint() -> None:
    source = (
        "from flask import Flask, Blueprint\n"
        "@app.route(\"/health\")\n"
        "def health(): pass\n"
        "@app.route(\"/users\", methods=[\"POST\", \"DELETE\", \"POST\"])\n"
        "def create_user(): pass\n"
        "@blueprint.get(\"/ready\")\n"
        "def ready(): pass\n"
    )

    routes = extract_structure(document(source, "python", "flask_app.py")).routes

    actual = [
        (route.path, route.methods, route.framework, route.handler)
        for route in routes
    ]
    assert actual == [
        ("/health", ("GET",), "flask", "health"),
        ("/users", ("POST", "DELETE"), "flask", "create_user"),
        ("/ready", ("GET",), "flask", "ready"),
    ]


@pytest.mark.parametrize("language", ["javascript", "typescript"])
def test_express_registrations_named_and_anonymous_handlers(language: str) -> None:
    source = (
        'app.get("/users", listUsers);\n'
        'router.post("/users", createUser);\n'
        'app.delete("/users/:id", (req, res) => {});\n'
    )

    routes = extract_structure(document(source, language, f"app.{language}")).routes

    actual = [
        (route.path, route.methods, route.framework, route.handler)
        for route in routes
    ]
    assert actual == [
        ("/users", ("GET",), "express", "listUsers"),
        ("/users", ("POST",), "express", "createUser"),
        ("/users/:id", ("DELETE",), "express", None),
    ]
    assert [(route.span.start_line, route.span.end_line) for route in routes] == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]


def test_spring_mappings_compose_class_prefix_and_normalize_methods() -> None:
    source = (
        '@RequestMapping("/api")\n'
        "class UserController {\n"
        '    @GetMapping(path = "/users")\n'
        "    public Object listUsers() { return null; }\n"
        '    @DeleteMapping("/users/{id}")\n'
        "    public void deleteUser() {}\n"
        "    @RequestMapping(\n"
        '        value = "/items",\n'
        "        method = {RequestMethod.POST, RequestMethod.PUT}\n"
        "    )\n"
        "    public void items() {}\n"
        "}\n"
    )

    routes = extract_structure(document(source, "java", "UserController.java")).routes

    actual = [
        (route.path, route.methods, route.handler, route.owner) for route in routes
    ]
    assert actual == [
        ("/api/users", ("GET",), "listUsers", "UserController"),
        ("/api/users/{id}", ("DELETE",), "deleteUser", "UserController"),
        ("/api/items", ("POST", "PUT"), "items", "UserController"),
    ]


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "python",
            '# @app.get("/fake")\ntext = \'@app.get("/also-fake")\'\n',
        ),
        (
            "javascript",
            '// app.get("/fake", handler);\n'
            'const text = "app.get(\'/also-fake\', handler)";\n',
        ),
        (
            "java",
            '// @GetMapping("/fake")\n'
            'class Plain { String text = "@GetMapping(\\\"/fake\\\")"; }\n',
        ),
    ],
)
def test_route_lookalikes_in_comments_and_strings_are_ignored(
    language: str, source: str
) -> None:
    assert extract_structure(document(source, language)).routes == ()


def test_unsupported_framework_and_dynamic_path_yield_no_routes() -> None:
    source = (
        "from django.urls import path\n"
        "path(\"users/\", users_view)\n"
        "@app.get(API_PREFIX + \"/users\")\n"
        "def users(): pass\n"
    )

    assert extract_structure(document(source, "python")).routes == ()


def test_fastapi_route_reaches_the_handler_chunk_end_to_end(tmp_path: Path) -> None:
    source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        '@app.get("/users/{id}")\n'
        "def get_user(id: int):\n"
        "    return {\"id\": id}\n"
    )
    (tmp_path / "api.py").write_text(source, encoding="utf-8")

    whole_file = load_source_documents(tmp_path)[0]
    chunks = chunk_documents([whole_file], chunk_size=55, chunk_overlap=5)
    enriched = enrich_chunks(chunks, extract_structure(whole_file))
    matching = [chunk for chunk in enriched if chunk.metadata["structural_routes"]]

    assert matching
    assert any("def get_user" in chunk.page_content for chunk in matching)
    assert matching[0].metadata["structural_routes"][0] == {
        "path": "/users/{id}",
        "methods": ["GET"],
        "framework": "fastapi",
        "handler": "get_user",
        "owner": "app",
        "start_line": 4,
        "end_line": 6,
    }
