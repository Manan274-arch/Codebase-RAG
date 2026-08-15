from pathlib import Path

import pytest
from backend.app import create_app
from fastapi.testclient import TestClient
from src.generation.context import ContextBundle, EvidenceItem
from src.generation.generator import GenerationResult
from src.indexing.qdrant_index import IndexBuildResult
from src.ingestion.acquisition import AcquiredRepository
from src.pipeline.codebase_rag import CodebaseAnswer, CodebaseRAGError


class FakeRAG:
    def __init__(self, repo_url: str, commit: str | None = None) -> None:
        self.repository = AcquiredRepository(
            repository_url=repo_url,
            commit_sha=commit or "a" * 40,
            checkout_path=Path("ignored-checkout"),
            reused_checkout=False,
        )
        self.source_file_count = 3
        self.chunk_count = 7
        self.index_result = IndexBuildResult(
            reused=False,
            point_count=7,
            embedding_seconds=0.1,
            upsert_seconds=0.1,
            fingerprint="fingerprint",
        )
        self.questions: list[str] = []
        self.closed = False
        self.ask_error = False

    def ask(self, question: str) -> CodebaseAnswer:
        if self.ask_error:
            raise CodebaseRAGError("question processing failed")
        self.questions.append(question)
        item = EvidenceItem(
            citation_id="C1",
            evidence_id="src/auth.py::2",
            source="src/auth.py",
            chunk_index=2,
            page_content=(
                "def authenticate(token: str) -> bool:\n    return bool(token)"
            ),
            start_line=10,
            end_line=11,
            origin="retrieved",
        )
        context = ContextBundle((item,), "[C1] src/auth.py:10-11\ncode")
        return CodebaseAnswer(
            question=question,
            generation=GenerationResult(
                "Authentication checks the token [C1].",
                ("C1",),
            ),
            evidence=(item,),
            context=context,
        )

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self) -> None:
        self.instances: list[FakeRAG] = []
        self.error: Exception | None = None

    def __call__(self, repo_url: str, commit: str | None) -> FakeRAG:
        if self.error is not None:
            raise self.error
        rag = FakeRAG(repo_url, commit)
        self.instances.append(rag)
        return rag


def load_repository(client: TestClient, *, commit: str | None = None) -> str:
    payload: dict[str, str | None] = {
        "repo_url": "https://github.com/example/project.git",
        "commit": commit,
    }
    response = client.post("/api/repositories", json=payload)
    assert response.status_code == 201
    return str(response.json()["repository_id"])


def test_health_endpoint() -> None:
    with TestClient(create_app(FakeFactory())) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_only_local_vite_origins() -> None:
    with TestClient(create_app(FakeFactory())) as client:
        allowed = client.options(
            "/api/repositories",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        rejected = client.options(
            "/api/repositories",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in rejected.headers


def test_loads_repository_once_and_returns_metadata() -> None:
    factory = FakeFactory()
    commit = "b" * 40
    with TestClient(create_app(factory)) as client:
        repository_id = load_repository(client, commit=commit)
        response = client.post(
            f"/api/repositories/{repository_id}/ask",
            json={"question": "Where is authentication handled?"},
        )

    assert len(factory.instances) == 1
    assert response.status_code == 200
    assert factory.instances[0].questions == ["Where is authentication handled?"]

    metadata_response = response
    assert metadata_response.json()["question"] == "Where is authentication handled?"


def test_repository_response_contains_prepared_metadata() -> None:
    factory = FakeFactory()
    commit = "b" * 40
    with TestClient(create_app(factory)) as client:
        response = client.post(
            "/api/repositories",
            json={
                "repo_url": "https://github.com/example/project.git",
                "commit": commit,
            },
        )

    assert response.status_code == 201
    assert response.json() == {
        "repository_id": response.json()["repository_id"],
        "repo_url": "https://github.com/example/project.git",
        "commit_sha": commit,
        "source_file_count": 3,
        "chunk_count": 7,
        "dense_index_status": "built",
    }


def test_serializes_answer_and_existing_citation_evidence() -> None:
    with TestClient(create_app(FakeFactory())) as client:
        repository_id = load_repository(client)
        response = client.post(
            f"/api/repositories/{repository_id}/ask",
            json={"question": "Where is authentication handled?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Where is authentication handled?",
        "answer": "Authentication checks the token [C1].",
        "citation_ids": ["C1"],
        "citations": [
            {
                "citation_id": "C1",
                "evidence_id": "src/auth.py::2",
                "source": "src/auth.py",
                "chunk_index": 2,
                "snippet": (
                    "def authenticate(token: str) -> bool:\n    return bool(token)"
                ),
                "start_line": 10,
                "end_line": 11,
                "origin": "retrieved",
            }
        ],
    }


@pytest.mark.parametrize("method", ["ask", "delete"])
def test_unknown_repository_returns_404(method: str) -> None:
    with TestClient(create_app(FakeFactory())) as client:
        if method == "ask":
            response = client.post(
                "/api/repositories/missing/ask",
                json={"question": "What happens?"},
            )
        else:
            response = client.delete("/api/repositories/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "repository session not found: missing"}


def test_delete_closes_repository_and_removes_session() -> None:
    factory = FakeFactory()
    with TestClient(create_app(factory)) as client:
        repository_id = load_repository(client)
        response = client.delete(f"/api/repositories/{repository_id}")
        missing = client.post(
            f"/api/repositories/{repository_id}/ask",
            json={"question": "Can I still ask?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "deleted",
        "repository_id": repository_id,
    }
    assert factory.instances[0].closed is True
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/repositories", {}),
        ("/api/repositories", {"repo_url": "", "commit": None}),
        ("/api/repositories", {"repo_url": "http://example.com/repo", "commit": None}),
        (
            "/api/repositories",
            {"repo_url": "https://example.com/repo", "commit": "short"},
        ),
        ("/api/repositories/anything/ask", {"question": "   "}),
    ],
)
def test_invalid_input_returns_422(path: str, payload: dict[str, object]) -> None:
    with TestClient(create_app(FakeFactory())) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 422


def test_repository_preparation_failure_is_safe_http_error() -> None:
    factory = FakeFactory()
    factory.error = CodebaseRAGError("repository indexing or pipeline setup failed")
    with TestClient(create_app(factory)) as client:
        response = client.post(
            "/api/repositories",
            json={"repo_url": "https://github.com/example/project.git"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "repository indexing or pipeline setup failed"
    }


def test_question_failure_is_safe_http_error() -> None:
    factory = FakeFactory()
    with TestClient(create_app(factory)) as client:
        repository_id = load_repository(client)
        factory.instances[0].ask_error = True
        response = client.post(
            f"/api/repositories/{repository_id}/ask",
            json={"question": "Trigger an error"},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "question processing failed"}


def test_application_shutdown_closes_all_retained_sessions() -> None:
    factory = FakeFactory()
    with TestClient(create_app(factory)) as client:
        load_repository(client)
        load_repository(client)
        assert all(not instance.closed for instance in factory.instances)

    assert len(factory.instances) == 2
    assert all(instance.closed for instance in factory.instances)
