"""Run and compare the brute-force dense and BM25 regression benchmarks."""

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


def run_comparison() -> tuple[RetrievalEvaluationResult, RetrievalEvaluationResult]:
    """Evaluate BM25 and the brute-force dense reference on one frozen corpus."""
    chunks = build_enriched_corpus(DEFAULT_FIXTURE_REPOSITORY)
    examples = load_evaluation_examples(DEFAULT_BENCHMARK)
    bm25 = evaluate_retriever(BM25Retriever(chunks), chunks, examples)
    dense = evaluate_retriever(DenseRetriever(chunks), chunks, examples)
    return bm25, dense


def format_comparison(
    bm25: RetrievalEvaluationResult, dense: RetrievalEvaluationResult
) -> str:
    """Format overall and category Hit Rate@1 comparisons."""
    lines = ["BM25 vs Dense", "", "Metric@K        BM25    Dense"]
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
                f"{getattr(dense.metrics[k], field):.4f}"
            )
    lines.extend(("", "Category HitRate@1", "Category       BM25    Dense"))
    for category in sorted(bm25.category_metrics):
        lines.append(
            f"{category:<15}"
            f"{bm25.category_metrics[category][1].hit_rate:<8.4f}"
            f"{dense.category_metrics[category][1].hit_rate:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    bm25, dense = run_comparison()
    print(format_report(bm25))
    print()
    print(format_report(dense).replace("BM25 Retrieval Baseline", "Dense Retrieval"))
    print()
    print(format_comparison(bm25, dense))


if __name__ == "__main__":
    main()
