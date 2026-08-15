"""Validate brute-force dense migration to persistent local Qdrant."""

from collections.abc import Sequence  # noqa: I001
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from src.retrieval import _torch_only  # noqa: F401
from evaluation.benchmark_v2 import (
    BENCHMARK_V2_QUERIES,
    BENCHMARK_V2_REPOSITORY,
    linked_context_coverage,
    load_and_validate_benchmark_v2,
)
from evaluation.brute_force_dense import DenseRetriever
from evaluation.metrics import (
    DocumentRetriever,
    RetrievalEvaluationExample,
    RetrievalEvaluationResult,
    evaluate_retriever,
    subset_evaluation_result,
)
from src.indexing.embeddings import SentenceTransformerBackend
from src.indexing.qdrant_index import IndexBuildResult, QdrantCodeIndex
from src.pipeline.corpus import build_enriched_corpus
from src.config import QdrantSettings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.candidate_union import CandidateUnionRetriever
from src.retrieval.qdrant_dense import QdrantDenseRetriever
from src.retrieval.relationship_expansion import RelationshipExpander
from src.retrieval.reranker import (
    CrossEncoderReranker,
    SentenceTransformerCrossEncoder,
)

CANDIDATE_KS = (10, 20, 25, 50)
FINAL_KS = (1, 3, 5, 10)


class TimedRetriever:
    def __init__(self, retriever: DocumentRetriever) -> None:
        self.retriever = retriever
        self.latencies: list[float] = []

    def retrieve(self, query: str, k: int = 10) -> Sequence[Any]:
        started = perf_counter()
        result = self.retriever.retrieve(query, k=k)
        self.latencies.append(perf_counter() - started)
        return result


class CachedRetriever:
    def __init__(self, retriever: DocumentRetriever) -> None:
        self.retriever = retriever
        self.cache: dict[str, tuple[int, list[Any]]] = {}

    def retrieve(self, query: str, k: int = 10) -> Sequence[Any]:
        cached = self.cache.get(query)
        if cached is None or cached[0] < k:
            cached = (k, list(self.retriever.retrieve(query, k=k)))
            self.cache[query] = cached
        return cached[1][:k]


@dataclass(frozen=True, slots=True)
class MigrationRun:
    corpus: tuple[Document, ...]
    examples: tuple[RetrievalEvaluationExample, ...]
    build: IndexBuildResult
    candidates: dict[str, RetrievalEvaluationResult]
    finals: dict[str, RetrievalEvaluationResult]
    latencies: dict[str, tuple[float, ...]]
    ann_recall_25: dict[str, float]
    final_runtimes: dict[str, float]
    real_hnsw: bool
    collection_status: str
    points_count: int
    indexed_vectors_count: int | None


def run_migration(*, rebuild: bool = False) -> MigrationRun:
    """Run exact correctness and local approximate-path checks on frozen v2."""
    corpus = tuple(build_enriched_corpus(BENCHMARK_V2_REPOSITORY))
    examples = load_and_validate_benchmark_v2(corpus, BENCHMARK_V2_QUERIES)
    encoder = SentenceTransformerBackend()
    reference = DenseRetriever(corpus, encoder=encoder)
    settings = QdrantSettings(
        path=Path(".qdrant") / "benchmark_v2",
        collection_name="benchmark_v2_chunks",
        search_mode="exact",
    )
    index = QdrantCodeIndex(settings)
    build = index.ensure(corpus, encoder, rebuild=rebuild)
    exact = QdrantDenseRetriever(index, encoder, search_mode="exact")
    local_approximate_path = QdrantDenseRetriever(
        index, encoder, search_mode="hnsw", hnsw_ef=128
    )
    timed = {
        "BruteForce": TimedRetriever(reference),
        "QdrantExact": TimedRetriever(exact),
        "LocalExactFalse": TimedRetriever(local_approximate_path),
    }
    for retriever in timed.values():
        retriever.retrieve(examples[0].query, k=25)
        retriever.latencies.clear()
    candidates = {
        name: evaluate_retriever(retriever, corpus, examples, ks=CANDIDATE_KS)
        for name, retriever in timed.items()
    }
    ann_recall = {
        name: _ann_recall_at_25(candidates["QdrantExact"], result)
        for name, result in candidates.items()
    }

    scorer = SentenceTransformerCrossEncoder()
    bm25 = BM25Retriever(corpus)
    final_inputs: dict[str, DocumentRetriever] = {
        "BruteForce": reference,
        "QdrantExact": exact,
        "LocalExactFalse": local_approximate_path,
    }
    finals: dict[str, RetrievalEvaluationResult] = {}
    final_runtimes: dict[str, float] = {}
    for name, dense in final_inputs.items():
        union = CandidateUnionRetriever(
            bm25, dense, bm25_depth=25, dense_depth=25
        )
        cached = CachedRetriever(CrossEncoderReranker(union, scorer=scorer))
        started = perf_counter()
        finals[name] = evaluate_retriever(
            RelationshipExpander(cached, corpus), corpus, examples, ks=FINAL_KS
        )
        final_runtimes[name] = perf_counter() - started
    info = index.client.get_collection(settings.collection_name)
    run = MigrationRun(
        corpus=corpus,
        examples=examples,
        build=build,
        candidates=candidates,
        finals=finals,
        latencies={name: tuple(item.latencies) for name, item in timed.items()},
        ann_recall_25=ann_recall,
        final_runtimes=final_runtimes,
        real_hnsw=index.supports_real_hnsw,
        collection_status=str(info.status),
        points_count=info.points_count or 0,
        indexed_vectors_count=info.indexed_vectors_count,
    )
    index.close()
    return run


def format_migration(run: MigrationRun) -> str:
    test_ids = frozenset(e.query_id for e in run.examples if e.split == "test")
    test_examples = tuple(e for e in run.examples if e.query_id in test_ids)
    lines = [
        "Qdrant Local Dense Migration",
        f"Index reused: {run.build.reused}",
        f"Points: {run.points_count}; status: {run.collection_status}; "
        f"indexed_vectors_count: {run.indexed_vectors_count}",
        f"Real HNSW active: {run.real_hnsw}",
        f"Build seconds: embeddings={run.build.embedding_seconds:.4f} "
        f"upsert={run.build.upsert_seconds:.4f}",
        "",
        "Held-out candidate metrics",
        "Backend          ANN@25  Recall@10/20/50       Hit@10/20/50  p50/p95 ms",
    ]
    for name, full in run.candidates.items():
        result = subset_evaluation_result(full, test_ids)
        recall = "/".join(f"{result.metrics[k].recall:.4f}" for k in (10, 20, 50))
        hit = "/".join(f"{result.metrics[k].hit_rate:.4f}" for k in (10, 20, 50))
        p50, p95 = _latency_ms(run.latencies[name])
        lines.append(
            f"{name:<17}{run.ann_recall_25[name]:<8.4f}{recall:<22}"
            f"{hit:<17}{p50:.3f}/{p95:.3f}"
        )
    lines.extend(("", "Held-out final expanded metrics"))
    lines.append(
        "Backend          Hit@1 Hit@3 Recall@5 MRR@5 "
        "nDCG@5 nDCG@10 Coverage@5 Runtime"
    )
    for name, full in run.finals.items():
        result = subset_evaluation_result(full, test_ids)
        coverage, _ = linked_context_coverage(
            result, test_examples, run.corpus, k=5
        )
        lines.append(
            f"{name:<17}{result.metrics[1].hit_rate:<6.4f}"
            f"{result.metrics[3].hit_rate:<6.4f}{result.metrics[5].recall:<9.4f}"
            f"{result.metrics[5].mrr:<6.4f}{result.metrics[5].ndcg:<7.4f}"
            f"{result.metrics[10].ndcg:<8.4f}{coverage:<11.4f}"
            f"{run.final_runtimes[name]:.2f}s"
        )
    lines.extend(
        (
            "",
            "LocalExactFalse is not an HNSW benchmark: qdrant-client local mode "
            "performs the same NumPy full scan and ignores search_params.",
            "Selected dense backend: persistent local Qdrant exact search.",
        )
    )
    return "\n".join(lines)


def _ann_recall_at_25(
    exact: RetrievalEvaluationResult, candidate: RetrievalEvaluationResult
) -> float:
    by_id = {query.query_id: query for query in candidate.queries}
    recalls = []
    for query in exact.queries:
        expected = frozenset(query.retrieved_chunk_ids[:25])
        actual = frozenset(by_id[query.query_id].retrieved_chunk_ids[:25])
        recalls.append(len(expected & actual) / len(expected))
    return sum(recalls) / len(recalls)


def _latency_ms(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return median(ordered) * 1000, ordered[p95_index] * 1000


def main() -> None:
    print(format_migration(run_migration()))


if __name__ == "__main__":
    main()
