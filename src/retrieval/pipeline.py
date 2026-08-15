"""Composition of the frozen production retrieval path."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from src.indexing.embeddings import EmbeddingBackend
from src.indexing.qdrant_index import QdrantCodeIndex
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.candidate_union import CandidateUnionRetriever
from src.retrieval.qdrant_dense import QdrantDenseRetriever
from src.retrieval.relationship_expansion import (
    ExpandedSearchResult,
    RelationshipExpander,
)
from src.retrieval.reranker import CrossEncoderReranker, PairScorer

PRODUCTION_BRANCH_DEPTH = 25


@dataclass(frozen=True, slots=True)
class RetrievalPipeline:
    """Production components with final query access through ``retrieve``."""

    bm25: BM25Retriever
    dense: QdrantDenseRetriever
    union: CandidateUnionRetriever
    reranker: CrossEncoderReranker
    final: RelationshipExpander

    def retrieve(self, query: str, k: int = 10) -> list[ExpandedSearchResult]:
        """Return final reranked and relationship-expanded results."""
        return list(self.final.retrieve(query, k=k))


def build_retrieval_pipeline(
    corpus: Sequence[Document],
    index: QdrantCodeIndex,
    encoder: EmbeddingBackend,
    *,
    scorer: PairScorer | None = None,
) -> RetrievalPipeline:
    """Compose production retrieval around an already built, validated index."""
    bm25 = BM25Retriever(corpus)
    dense = QdrantDenseRetriever(index, encoder)
    union = CandidateUnionRetriever(
        bm25,
        dense,
        bm25_depth=PRODUCTION_BRANCH_DEPTH,
        dense_depth=PRODUCTION_BRANCH_DEPTH,
    )
    reranker = CrossEncoderReranker(union, scorer=scorer)
    final = RelationshipExpander(reranker, corpus)
    return RetrievalPipeline(bm25, dense, union, reranker, final)
