"""End-to-end orchestration of the production Codebase RAG components."""

import hashlib  # noqa: I001
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType

from src.retrieval import _torch_only  # noqa: F401
from src.config import QdrantSettings
from src.generation.context import ContextBundle, EvidenceItem, build_context
from src.generation.generator import GenerationResult, TextGenerator, generate_answer
from src.generation.groq import GroqTextGenerator
from src.indexing.embeddings import EmbeddingBackend, SentenceTransformerBackend
from src.indexing.qdrant_index import IndexBuildResult, QdrantCodeIndex
from src.ingestion.pipeline import build_enriched_corpus
from src.ingestion.repository import find_source_files
from src.repository_acquisition import (
    DEFAULT_REPOSITORY_CACHE,
    AcquiredRepository,
    acquire_repository,
)
from src.retrieval.pipeline import RetrievalPipeline, build_retrieval_pipeline
from src.retrieval.reranker import PairScorer


class CodebaseRAGError(RuntimeError):
    """Raised when repository preparation or question processing fails."""


@dataclass(frozen=True, slots=True)
class CodebaseAnswer:
    """One generated answer with its validated evidence projection."""

    question: str
    generation: GenerationResult
    evidence: tuple[EvidenceItem, ...]
    context: ContextBundle

    @property
    def answer(self) -> str:
        return self.generation.answer

    @property
    def citation_ids(self) -> tuple[str, ...]:
        return self.generation.citation_ids


class CodebaseRAG:
    """Prepared repository state reusable across multiple questions."""

    def __init__(
        self,
        repository: AcquiredRepository,
        *,
        source_file_count: int,
        chunk_count: int,
        index: QdrantCodeIndex,
        index_result: IndexBuildResult,
        retrieval: RetrievalPipeline,
        generator: TextGenerator,
    ) -> None:
        self.repository = repository
        self.source_file_count = source_file_count
        self.chunk_count = chunk_count
        self.index_result = index_result
        self._index = index
        self._retrieval = retrieval
        self._generator = generator
        self._closed = False

    @classmethod
    def from_repository_url(
        cls,
        repository_url: str,
        *,
        commit: str | None = None,
        cache_dir: Path = DEFAULT_REPOSITORY_CACHE,
        qdrant_settings: QdrantSettings | None = None,
        encoder: EmbeddingBackend | None = None,
        scorer: PairScorer | None = None,
        generator: TextGenerator | None = None,
    ) -> "CodebaseRAG":
        """Acquire and prepare one immutable repository for repeated questions."""
        repository = acquire_repository(
            repository_url,
            commit,
            cache_dir=cache_dir,
        )
        source_files = find_source_files(repository.checkout_path)
        if not source_files:
            raise CodebaseRAGError("repository contains no supported source files")
        try:
            corpus = build_enriched_corpus(repository.checkout_path)
        except Exception as error:
            raise CodebaseRAGError("repository ingestion failed") from error
        if not corpus:
            raise CodebaseRAGError("repository produced no code chunks")

        settings = qdrant_settings or QdrantSettings.from_environment()
        settings = replace(
            settings,
            collection_name=_repository_collection_name(settings, repository),
        )
        selected_encoder = encoder or SentenceTransformerBackend(
            settings.embedding_model
        )
        index = QdrantCodeIndex(settings)
        try:
            index_result = index.ensure(corpus, selected_encoder)
            retrieval = build_retrieval_pipeline(
                corpus,
                index,
                selected_encoder,
                scorer=scorer,
            )
            selected_generator = generator or GroqTextGenerator()
        except Exception as error:
            index.close()
            raise CodebaseRAGError(
                "repository indexing or pipeline setup failed"
            ) from error

        return cls(
            repository,
            source_file_count=len(source_files),
            chunk_count=len(corpus),
            index=index,
            index_result=index_result,
            retrieval=retrieval,
            generator=selected_generator,
        )

    def ask(
        self,
        question: str,
        *,
        retrieval_k: int = 10,
        max_context_chars: int | None = None,
    ) -> CodebaseAnswer:
        """Retrieve once and generate a citation-validated answer."""
        if self._closed:
            raise CodebaseRAGError("CodebaseRAG is closed")
        try:
            retrieved = self._retrieval.retrieve(question, k=retrieval_k)
            context = build_context(retrieved, max_chars=max_context_chars)
            generation = generate_answer(question, context, self._generator)
        except Exception as error:
            raise CodebaseRAGError("question processing failed") from error
        evidence_by_citation = {
            item.citation_id: item for item in context.evidence
        }
        evidence = tuple(
            evidence_by_citation[citation_id]
            for citation_id in generation.citation_ids
        )
        return CodebaseAnswer(question, generation, evidence, context)

    def close(self) -> None:
        """Release the persistent local Qdrant client lock."""
        if not self._closed:
            self._index.close()
            self._closed = True

    def __enter__(self) -> "CodebaseRAG":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _repository_collection_name(
    settings: QdrantSettings,
    repository: AcquiredRepository,
) -> str:
    material = f"{repository.repository_url}\0{repository.commit_sha}"
    identity = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{settings.collection_name}_repo_{identity}"
