"""Run BM25, dense, and RRF on the frozen retrieval benchmark."""

from src.retrieval import _torch_only  # noqa: F401, I001
from src.pipeline.corpus import build_enriched_corpus
from src.retrieval.bm25 import BM25Retriever
from evaluation.brute_force_dense import DenseRetriever
from evaluation.runners.evaluate_bm25 import (
    DEFAULT_BENCHMARK,
    DEFAULT_FIXTURE_REPOSITORY,
    format_report,
)
from evaluation.metrics import (
    RetrievalEvaluationResult,
    evaluate_retriever,
    load_evaluation_examples,
)
from evaluation.rrf import HybridRetriever


def run_comparison() -> tuple[
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
]:
    """Evaluate all three retrievers on the identical frozen corpus and labels."""
    chunks = build_enriched_corpus(DEFAULT_FIXTURE_REPOSITORY)
    examples = load_evaluation_examples(DEFAULT_BENCHMARK)
    bm25_retriever = BM25Retriever(chunks)
    dense_retriever = DenseRetriever(chunks)
    hybrid_retriever = HybridRetriever(bm25_retriever, dense_retriever)
    return (
        evaluate_retriever(bm25_retriever, chunks, examples),
        evaluate_retriever(dense_retriever, chunks, examples),
        evaluate_retriever(hybrid_retriever, chunks, examples),
    )


def format_three_way_comparison(
    bm25: RetrievalEvaluationResult,
    dense: RetrievalEvaluationResult,
    rrf: RetrievalEvaluationResult,
) -> str:
    """Format direct overall and category comparisons for all retrievers."""
    lines = ["BM25 vs Dense vs RRF", "", "Metric@K        BM25    Dense   RRF"]
    fields = (
        ("HitRate", "hit_rate"),
        ("Recall", "recall"),
        ("MRR", "mrr"),
        ("nDCG", "ndcg"),
    )
    for k in bm25.ks:
        for label, field in fields:
            lines.append(
                f"{label}@{k:<9}"
                f"{getattr(bm25.metrics[k], field):<8.4f}"
                f"{getattr(dense.metrics[k], field):<8.4f}"
                f"{getattr(rrf.metrics[k], field):.4f}"
            )
    lines.extend(("", "Category HitRate@1", "Category       BM25    Dense   RRF"))
    for category in sorted(bm25.category_metrics):
        lines.append(
            f"{category:<15}"
            f"{bm25.category_metrics[category][1].hit_rate:<8.4f}"
            f"{dense.category_metrics[category][1].hit_rate:<8.4f}"
            f"{rrf.category_metrics[category][1].hit_rate:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    bm25, dense, rrf = run_comparison()
    print(format_report(bm25))
    print()
    print(format_report(dense).replace("BM25 Retrieval Baseline", "Dense Retrieval"))
    print()
    print(format_report(rrf).replace("BM25 Retrieval Baseline", "Hybrid RRF"))
    print()
    print(format_three_way_comparison(bm25, dense, rrf))


if __name__ == "__main__":
    main()
