"""Raw-source-only BM25 lexical retrieval baseline."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    """One original corpus document and its BM25 score."""

    document: Document
    score: float


class BM25Retriever:
    """Index and search only the raw ``page_content`` of code chunks."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = tuple(documents)
        tokenized_corpus = [
            tokenize(document.page_content) for document in self._documents
        ]
        self._index = (
            BM25Okapi(tokenized_corpus)
            if tokenized_corpus and any(tokenized_corpus)
            else None
        )

    def retrieve(self, query: str, k: int = 10) -> list[BM25SearchResult]:
        """Return up to ``k`` results ordered by score, then corpus position."""
        if k < 0:
            raise ValueError("k must be non-negative")
        query_tokens = tokenize(query)
        if k == 0 or not query_tokens or not self._documents:
            return []

        if self._index is None:
            return [
                BM25SearchResult(document=document, score=0.0)
                for document in self._documents[:k]
            ]

        scores = self._index.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [
            BM25SearchResult(
                document=self._documents[index],
                score=float(score),
            )
            for index, score in ranked[: min(k, len(self._documents))]
        ]


def tokenize(text: str) -> list[str]:
    """Lowercase and extract ASCII alphanumeric/underscore tokens."""
    return _TOKEN_PATTERN.findall(text.casefold())
