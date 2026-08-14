"""Evaluate the local cross-encoder after RRF on the frozen benchmark."""

from src.retrieval import _torch_only  # noqa: F401, I001
from src.ingestion.pipeline import build_enriched_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.evaluate_bm25 import (
    DEFAULT_BENCHMARK,
    DEFAULT_FIXTURE_REPOSITORY,
    format_report,
)
from src.retrieval.evaluation import (
    QueryEvaluation,
    RetrievalEvaluationResult,
    evaluate_retriever,
    load_evaluation_examples,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def run_comparison() -> tuple[
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
]:
    """Evaluate dense, RRF, and RRF plus the default cross-encoder."""
    chunks = build_enriched_corpus(DEFAULT_FIXTURE_REPOSITORY)
    examples = load_evaluation_examples(DEFAULT_BENCHMARK)
    dense_retriever = DenseRetriever(chunks)
    hybrid_retriever = HybridRetriever(BM25Retriever(chunks), dense_retriever)
    reranker = CrossEncoderReranker(hybrid_retriever)
    return (
        evaluate_retriever(dense_retriever, chunks, examples),
        evaluate_retriever(hybrid_retriever, chunks, examples),
        evaluate_retriever(reranker, chunks, examples),
    )


def changed_relevant_ranks(
    rrf: RetrievalEvaluationResult,
    reranked: RetrievalEvaluationResult,
) -> tuple[tuple[QueryEvaluation, QueryEvaluation], ...]:
    """Return aligned query results whose first relevant rank changed."""
    return tuple(
        (before, after)
        for before, after in zip(rrf.queries, reranked.queries, strict=True)
        if before.first_relevant_rank != after.first_relevant_rank
    )


def format_comparison(
    dense: RetrievalEvaluationResult,
    rrf: RetrievalEvaluationResult,
    reranked: RetrievalEvaluationResult,
) -> str:
    """Format direct metrics, category Hit@1, and changed relevant ranks."""
    lines = [
        "Dense vs RRF vs RRF + Cross-Encoder",
        "",
        "Metric@K        Dense   RRF     Reranked",
    ]
    fields = (
        ("HitRate", "hit_rate"),
        ("Recall", "recall"),
        ("MRR", "mrr"),
        ("nDCG", "ndcg"),
    )
    for k in dense.ks:
        for label, field in fields:
            lines.append(
                f"{label}@{k:<9}"
                f"{getattr(dense.metrics[k], field):<8.4f}"
                f"{getattr(rrf.metrics[k], field):<8.4f}"
                f"{getattr(reranked.metrics[k], field):.4f}"
            )
    lines.extend(
        ("", "Category HitRate@1", "Category       Dense   RRF     Reranked")
    )
    for category in sorted(dense.category_metrics):
        lines.append(
            f"{category:<15}"
            f"{dense.category_metrics[category][1].hit_rate:<8.4f}"
            f"{rrf.category_metrics[category][1].hit_rate:<8.4f}"
            f"{reranked.category_metrics[category][1].hit_rate:.4f}"
        )
    lines.extend(("", "Changed first-relevant ranks"))
    changes = changed_relevant_ranks(rrf, reranked)
    if not changes:
        lines.append("none")
    for before, after in changes:
        lines.append(
            f"{before.query_id} ({before.category}): "
            f"{before.first_relevant_rank} -> {after.first_relevant_rank}"
        )
    return "\n".join(lines)


def main() -> None:
    dense, rrf, reranked = run_comparison()
    print(format_report(rrf).replace("BM25 Retrieval Baseline", "Hybrid RRF"))
    print()
    print(
        format_report(reranked).replace(
            "BM25 Retrieval Baseline", "RRF + Cross-Encoder"
        )
    )
    print()
    print(format_comparison(dense, rrf, reranked))


if __name__ == "__main__":
    main()
