from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from evaluation.brute_force_dense import DenseRetriever
from evaluation.rrf import HybridRetriever
from langchain_core.documents import Document
from src.enrichment.representation import (
    RAW_CONTENT_METADATA_KEY,
    enrich_retrieval_content,
    render_retrieval_content,
)
from src.pipeline.corpus import build_enriched_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.contracts import canonical_chunk_id


class FakeEncoder:
    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> npt.NDArray[np.float32]:
        assert normalize_embeddings
        return np.asarray(
            [[1.0, float("GET /users" in text)] for text in texts],
            dtype=np.float32,
        )


def structured_document() -> Document:
    return Document(
        page_content="def get_user():\n    return service.load()\n",
        metadata={
            "source": "backend/users.py",
            "language": "python",
            "chunk_index": 7,
            "start_line": 10,
            "end_line": 12,
            "structural_definitions": [
                {
                    "name": "get_user",
                    "qualified_name": "Users.get_user",
                    "kind": "method",
                    "signature": "def get_user()",
                    "start_line": 10,
                    "end_line": 12,
                }
            ],
            "structural_imports": [
                {
                    "source": "from app import service",
                    "items": ["service"],
                    "alias": None,
                    "is_wildcard": False,
                    "start_line": 1,
                    "end_line": 1,
                }
            ],
            "structural_routes": [
                {
                    "path": "/users/{id}",
                    "methods": ["GET"],
                    "framework": "fastapi",
                    "handler": "get_user",
                    "owner": "router",
                    "start_line": 9,
                    "end_line": 12,
                }
            ],
            "structural_http_calls": [
                {
                    "method": "GET",
                    "target": "/profiles/{id}",
                    "client": "httpx",
                    "caller": "get_user",
                    "start_line": 11,
                    "end_line": 11,
                }
            ],
            "related_route_chunks": [
                {
                    "source": "opaque.py",
                    "chunk_index": 3,
                    "start_line": 1,
                    "end_line": 2,
                }
            ],
            "related_http_call_chunks": [],
            "internal_secret": {"do_not": "serialize me"},
        },
        id="original-id",
    )


def test_exact_structural_representation_and_raw_recovery() -> None:
    original = structured_document()

    enriched = enrich_retrieval_content(original)

    assert enriched.page_content == (
        "Source: backend/users.py\n\n"
        "Definitions:\n"
        "- Users.get_user | Kind: method | Signature: def get_user()\n\n"
        "Imports:\n"
        "- from app import service | Items: service\n\n"
        "Routes:\n"
        "- GET /users/{id} | Framework: fastapi | Handler: get_user | Owner: router\n\n"
        "Outbound HTTP Calls:\n"
        "- GET /profiles/{id} | Client: httpx | Caller: get_user\n\n"
        "Code:\n"
        "def get_user():\n    return service.load()\n"
    )
    assert enriched.metadata[RAW_CONTENT_METADATA_KEY] == original.page_content
    assert enriched.id == original.id
    assert original.metadata.get(RAW_CONTENT_METADATA_KEY) is None


def test_empty_sections_are_omitted_without_serializing_bookkeeping() -> None:
    metadata = {
        "source": "empty.py",
        "chunk_index": 42,
        "structural_definitions": [],
        "related_route_chunks": [{"source": "routes.py", "chunk_index": 9}],
        "arbitrary": {"secret": "value"},
    }

    rendered = render_retrieval_content("", metadata)

    assert rendered == "Source: empty.py\n\nCode:\n"
    assert "Definitions:" not in rendered
    assert "42" not in rendered
    assert "routes.py::9" not in rendered
    assert "secret" not in rendered


def test_rendering_is_deterministic_and_repeated_enrichment_is_idempotent() -> None:
    original = structured_document()

    first = enrich_retrieval_content(original)
    second = enrich_retrieval_content(first)

    assert first.page_content == second.page_content
    assert (
        first.metadata[RAW_CONTENT_METADATA_KEY]
        == second.metadata[RAW_CONTENT_METADATA_KEY]
    )
    assert second.page_content.count("Definitions:") == 1
    assert second.page_content.count("Code:") == 1


def test_identity_structure_and_relationship_metadata_remain_intact() -> None:
    original = structured_document()
    before_identity = canonical_chunk_id(original)

    enriched = enrich_retrieval_content(original)

    assert canonical_chunk_id(enriched) == before_identity == "backend/users.py::7"
    for key in (
        "structural_definitions",
        "structural_imports",
        "structural_routes",
        "structural_http_calls",
        "related_route_chunks",
        "related_http_call_chunks",
    ):
        assert enriched.metadata[key] == original.metadata[key]


def test_shared_representation_works_with_all_retrievers() -> None:
    route = enrich_retrieval_content(structured_document())
    other = enrich_retrieval_content(
        Document(
            page_content="def unrelated(): pass",
            metadata={"source": "other.py", "chunk_index": 0},
        )
    )
    documents = [route, other]
    bm25 = BM25Retriever(documents)
    dense = DenseRetriever(documents, encoder=FakeEncoder())
    hybrid = HybridRetriever(bm25, dense, candidate_depth=2)

    assert bm25.retrieve("GET /users", k=1)[0].document is route
    assert dense.retrieve("GET /users", k=1)[0].document is route
    assert hybrid.retrieve("GET /users", k=1)[0].document is route


def test_real_multilanguage_pipeline_renders_and_preserves_links(
    tmp_path: Path,
) -> None:
    (tmp_path / "backend.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        '@app.get("/users/{id}")\n'
        "def get_user(id: str):\n"
        "    return id\n",
        encoding="utf-8",
    )
    (tmp_path / "frontend.ts").write_text(
        'import axios from "axios";\n'
        "export async function loadUser(id: string) {\n"
        "    return axios.get(`/users/${id}`);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "Health.java").write_text(
        'class Health { public String status() { return "ready"; } }\n',
        encoding="utf-8",
    )

    corpus = build_enriched_corpus(tmp_path)
    backend = next(item for item in corpus if item.metadata["source"] == "backend.py")
    frontend = next(item for item in corpus if item.metadata["source"] == "frontend.ts")
    java = next(item for item in corpus if item.metadata["source"] == "Health.java")

    assert "Routes:\n- GET /users/{id}" in backend.page_content
    assert "Outbound HTTP Calls:\n- GET /users/{id}" in frontend.page_content
    assert "Definitions:" in java.page_content
    assert backend.metadata["related_http_call_chunks"]
    assert frontend.metadata["related_route_chunks"]
    assert backend.metadata[RAW_CONTENT_METADATA_KEY].startswith("from fastapi")
    assert frontend.metadata[RAW_CONTENT_METADATA_KEY].startswith("import axios")
