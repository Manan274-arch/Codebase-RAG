"""FastAPI application wrapping reusable Codebase RAG sessions."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from src.ingestion.acquisition import RepositoryAcquisitionError
from src.pipeline.codebase_rag import CodebaseAnswer, CodebaseRAGError

from backend.models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    DeleteResponse,
    HealthResponse,
    RepositoryRequest,
    RepositoryResponse,
)
from backend.service import (
    RepositoryFactory,
    RepositoryRAG,
    RepositoryService,
    UnknownRepositoryError,
    load_codebase_rag,
)

_LOCAL_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def create_app(factory: RepositoryFactory = load_codebase_rag) -> FastAPI:
    """Create one application with its own process-local repository registry."""
    service = RepositoryService(factory)
    allowed_origins = list(_LOCAL_FRONTEND_ORIGINS)
    frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
    if frontend_url and frontend_url not in allowed_origins:
        allowed_origins.append(frontend_url)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.repository_service = service
        try:
            yield
        finally:
            service.close_all()

    application = FastAPI(title="Codebase RAG Demo API", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.post(
        "/api/repositories",
        response_model=RepositoryResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_repository(request: RepositoryRequest) -> RepositoryResponse:
        try:
            repository_id, rag = service.create(request.repo_url, request.commit)
        except (ValueError, RepositoryAcquisitionError, CodebaseRAGError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        return _repository_response(repository_id, rag)

    @application.post(
        "/api/repositories/{repository_id}/ask",
        response_model=AskResponse,
    )
    def ask_repository(repository_id: str, request: AskRequest) -> AskResponse:
        try:
            answer = service.ask(repository_id, request.question)
        except UnknownRepositoryError as error:
            raise _not_found(repository_id) from error
        except CodebaseRAGError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(error),
            ) from error
        return _answer_response(answer)

    @application.delete(
        "/api/repositories/{repository_id}",
        response_model=DeleteResponse,
    )
    def delete_repository(repository_id: str) -> DeleteResponse:
        try:
            service.delete(repository_id)
        except UnknownRepositoryError as error:
            raise _not_found(repository_id) from error
        return DeleteResponse(repository_id=repository_id)

    return application


def _repository_response(
    repository_id: str,
    rag: RepositoryRAG,
) -> RepositoryResponse:
    return RepositoryResponse(
        repository_id=repository_id,
        repo_url=rag.repository.repository_url,
        commit_sha=rag.repository.commit_sha,
        source_file_count=rag.source_file_count,
        chunk_count=rag.chunk_count,
        dense_index_status="reused" if rag.index_result.reused else "built",
    )


def _answer_response(answer: CodebaseAnswer) -> AskResponse:
    citations = [
        CitationResponse(
            citation_id=item.citation_id,
            evidence_id=item.evidence_id,
            source=item.source,
            chunk_index=item.chunk_index,
            snippet=item.page_content,
            start_line=item.start_line,
            end_line=item.end_line,
            origin=item.origin,
        )
        for item in answer.evidence
    ]
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        citation_ids=list(answer.citation_ids),
        citations=citations,
    )


def _not_found(repository_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"repository session not found: {repository_id}",
    )


app = create_app()
