"""Run historical retrieval stages on frozen Retrieval Benchmark v2."""

from dataclasses import dataclass  # noqa: I001
from time import perf_counter

from langchain_core.documents import Document

from src.retrieval import _torch_only  # noqa: F401
from src.pipeline.corpus import build_enriched_corpus
from evaluation.benchmark_v2 import (
    BENCHMARK_V2_QUERIES,
    BENCHMARK_V2_REPOSITORY,
    linked_context_coverage,
    load_and_validate_benchmark_v2,
)
from src.retrieval.bm25 import BM25Retriever
from evaluation.brute_force_dense import DenseRetriever
from evaluation.metrics import (
    DocumentRetriever,
    RetrievalEvaluationExample,
    RetrievalEvaluationResult,
    evaluate_retriever,
    subset_evaluation_result,
)
from evaluation.rrf import HybridRetriever
from src.retrieval.relationship_expansion import RelationshipExpander
from src.retrieval.reranker import CrossEncoderReranker


@dataclass(frozen=True, slots=True)
class BenchmarkV2Run:
    corpus: tuple[Document, ...]
    examples: tuple[RetrievalEvaluationExample, ...]
    results: dict[str, RetrievalEvaluationResult]
    runtimes: dict[str, float]


def run_benchmark_v2() -> BenchmarkV2Run:
    """Evaluate the historical retrieval stages once across both stored splits."""
    corpus = tuple(build_enriched_corpus(BENCHMARK_V2_REPOSITORY))
    examples = load_and_validate_benchmark_v2(corpus, BENCHMARK_V2_QUERIES)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    rrf = HybridRetriever(bm25, dense)
    reranker = CrossEncoderReranker(rrf)
    expansion = RelationshipExpander(reranker, corpus)
    stages: tuple[tuple[str, DocumentRetriever], ...] = (
        ("BM25", bm25),
        ("Dense", dense),
        ("RRF", rrf),
        ("Reranker", reranker),
        ("Expansion", expansion),
    )
    results: dict[str, RetrievalEvaluationResult] = {}
    runtimes: dict[str, float] = {}
    for name, retriever in stages:
        started = perf_counter()
        results[name] = evaluate_retriever(retriever, corpus, examples)
        runtimes[name] = perf_counter() - started
    return BenchmarkV2Run(corpus, examples, results, runtimes)


def format_benchmark_v2(run: BenchmarkV2Run) -> str:
    """Format split, category, coverage, runtime, and failure diagnostics."""
    lines = [
        "Retrieval Benchmark v2",
        f"Corpus chunks: {len(run.corpus)}",
        f"Queries: {len(run.examples)}",
    ]
    for split in ("dev", "test"):
        ids = frozenset(
            example.query_id for example in run.examples if example.split == split
        )
        split_results = {
            name: subset_evaluation_result(result, ids)
            for name, result in run.results.items()
        }
        lines.extend(("", f"Split: {split}", _overall_table(split_results)))
        if split == "test":
            lines.extend(("", "Held-out categories", _category_table(split_results)))
            lines.extend(("", "Linked Context Coverage", _coverage_table(run, ids)))
            lines.extend(("", "Representative lowest nDCG@5 expansion queries"))
            lines.extend(_failure_lines(split_results["Expansion"], run.examples))
    lines.extend(("", "Stage runtime (seconds)"))
    lines.extend(f"{name}: {seconds:.2f}" for name, seconds in run.runtimes.items())
    return "\n".join(lines)


def _overall_table(results: dict[str, RetrievalEvaluationResult]) -> str:
    lines = [
        "Stage       "
        "Hit@1/3/5/10                 Recall@1/3/5/10              "
        "MRR@1/3/5/10                nDCG@1/3/5/10"
    ]
    for name, result in results.items():
        groups = []
        for field in ("hit_rate", "recall", "mrr", "ndcg"):
            groups.append(
                "/".join(f"{getattr(result.metrics[k], field):.4f}" for k in result.ks)
            )
        lines.append(
            f"{name:<12}" + "  ".join(f"{group:<27}" for group in groups).rstrip()
        )
    return "\n".join(lines)


def _category_table(results: dict[str, RetrievalEvaluationResult]) -> str:
    lines = ["Stage       Category      Hit@1   Hit@3   MRR@5   nDCG@5"]
    for name, result in results.items():
        for category, metrics in result.category_metrics.items():
            lines.append(
                f"{name:<12}{category:<14}"
                f"{metrics[1].hit_rate:<8.4f}"
                f"{metrics[3].hit_rate:<8.4f}"
                f"{metrics[5].mrr:<8.4f}"
                f"{metrics[5].ndcg:.4f}"
            )
    return "\n".join(lines)


def _coverage_table(run: BenchmarkV2Run, test_ids: frozenset[str]) -> str:
    test_examples = tuple(
        example for example in run.examples if example.query_id in test_ids
    )
    lines = ["Stage       @1      @3      @5      @10    Eligible"]
    for name, full_result in run.results.items():
        result = subset_evaluation_result(full_result, test_ids)
        coverages = [
            linked_context_coverage(result, test_examples, run.corpus, k=cutoff)
            for cutoff in (1, 3, 5, 10)
        ]
        eligible = coverages[0][1]
        lines.append(
            f"{name:<12}"
            + "".join(f"{coverage:<8.4f}" for coverage, _ in coverages)
            + str(eligible)
        )
    return "\n".join(lines)


def _failure_lines(
    result: RetrievalEvaluationResult,
    examples: tuple[RetrievalEvaluationExample, ...],
) -> list[str]:
    examples_by_id = {example.query_id: example for example in examples}
    lowest = sorted(
        result.queries,
        key=lambda query: (query.metrics[5].ndcg, query.query_id),
    )[:15]
    lines: list[str] = []
    for query in lowest:
        example = examples_by_id[query.query_id]
        primary = sorted(
            chunk_id
            for chunk_id, grade in (example.relevance_grades or {}).items()
            if grade == 2
        )
        lines.append(
            f"{query.query_id}|{query.category}|nDCG5={query.metrics[5].ndcg:.4f}"
            f"|expected={','.join(primary)}"
            f"|top3={','.join(query.retrieved_chunk_ids[:3])}"
        )
    return lines


def main() -> None:
    print(format_benchmark_v2(run_benchmark_v2()))


if __name__ == "__main__":
    main()
