"""Process-local lifecycle management for prepared Codebase RAG sessions."""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from src.indexing.qdrant_index import IndexBuildResult
from src.ingestion.acquisition import AcquiredRepository
from src.pipeline.codebase_rag import CodebaseAnswer, CodebaseRAG


class RepositoryRAG(Protocol):
    repository: AcquiredRepository
    source_file_count: int
    chunk_count: int
    index_result: IndexBuildResult

    def ask(self, question: str) -> CodebaseAnswer: ...

    def close(self) -> None: ...


RepositoryFactory = Callable[[str, str | None], RepositoryRAG]


class UnknownRepositoryError(KeyError):
    """Raised when a process-local repository session does not exist."""


@dataclass(slots=True)
class RepositorySession:
    rag: RepositoryRAG
    lock: threading.Lock = field(default_factory=threading.Lock)


def load_codebase_rag(repo_url: str, commit: str | None) -> RepositoryRAG:
    """Construct the existing production orchestration without duplicating it."""
    return CodebaseRAG.from_repository_url(repo_url, commit=commit)


class RepositoryService:
    """Retain prepared repositories for the lifetime of this application process."""

    def __init__(self, factory: RepositoryFactory = load_codebase_rag) -> None:
        self._factory = factory
        self._sessions: dict[str, RepositorySession] = {}
        self._lock = threading.Lock()

    def create(self, repo_url: str, commit: str | None) -> tuple[str, RepositoryRAG]:
        rag = self._factory(repo_url, commit)
        repository_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[repository_id] = RepositorySession(rag)
        return repository_id, rag

    def ask(self, repository_id: str, question: str) -> CodebaseAnswer:
        session = self._locked_session(repository_id)
        try:
            return session.rag.ask(question)
        finally:
            session.lock.release()

    def delete(self, repository_id: str) -> None:
        with self._lock:
            session = self._sessions.get(repository_id)
            if session is None:
                raise UnknownRepositoryError(repository_id)
            session.lock.acquire()
            del self._sessions[repository_id]
        try:
            session.rag.close()
        finally:
            session.lock.release()

    def close_all(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        first_error: Exception | None = None
        for session in sessions:
            with session.lock:
                try:
                    session.rag.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error

    def _locked_session(self, repository_id: str) -> RepositorySession:
        with self._lock:
            session = self._sessions.get(repository_id)
            if session is None:
                raise UnknownRepositoryError(repository_id)
            session.lock.acquire()
            return session
