"""Evaluate post-reranking relationship expansion on the frozen benchmark."""

from src.retrieval import _torch_only  # noqa: F401, I001
from src.ingestion.pipeline import build_enriched_corpus
from src.retrieval.bm25 import BM25Retriever
from src.evaluation.brute_force_dense import DenseRetriever
from src.evaluation.runners.evaluate_bm25 import (
    DEFAULT_BENCHMARK,
    DEFAULT_FIXTURE_REPOSITORY,
    format_report,
)
from src.evaluation.metrics import (
    QueryEvaluation,
    RetrievalEvaluationResult,
    evaluate_retriever,
    load_evaluation_examples,
)
from src.evaluation.rrf import HybridRetriever
from src.retrieval.relationship_expansion import RelationshipExpander
from src.retrieval.reranker import CrossEncoderReranker


def run_comparison() -> tuple[
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
    RetrievalEvaluationResult,
]:
    """Evaluate dense, RRF, reranking, and bounded relationship expansion."""
    chunks = build_enriched_corpus(DEFAULT_FIXTURE_REPOSITORY)
    examples = load_evaluation_examples(DEFAULT_BENCHMARK)
    dense_retriever = DenseRetriever(chunks)
    hybrid_retriever = HybridRetriever(BM25Retriever(chunks), dense_retriever)
    reranker = CrossEncoderReranker(hybrid_retriever)
    expander = RelationshipExpander(reranker, chunks)
    return (
        evaluate_retriever(dense_retriever, chunks, examples),
        evaluate_retriever(hybrid_retriever, chunks, examples),
        evaluate_retriever(reranker, chunks, examples),
        evaluate_retriever(expander, chunks, examples),
    )


def format_comparison(
    dense: RetrievalEvaluationResult,
    rrf: RetrievalEvaluationResult,
    reranked: RetrievalEvaluationResult,
    expanded: RetrievalEvaluationResult,
) -> str:
    """Format direct aggregate, category, and relationship-query comparisons."""
    lines = [
        "Dense vs RRF vs Reranked vs Expanded",
        "",
        "Metric@K        Dense   RRF     Rerank  Expand",
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
                f"{getattr(reranked.metrics[k], field):<8.4f}"
                f"{getattr(expanded.metrics[k], field):.4f}"
            )
    lines.extend(
        (
            "",
            "Category HitRate@1",
            "Category       Dense   RRF     Rerank  Expand",
        )
    )
    for category in sorted(dense.category_metrics):
        lines.append(
            f"{category:<15}"
            f"{dense.category_metrics[category][1].hit_rate:<8.4f}"
            f"{rrf.category_metrics[category][1].hit_rate:<8.4f}"
            f"{reranked.category_metrics[category][1].hit_rate:<8.4f}"
            f"{expanded.category_metrics[category][1].hit_rate:.4f}"
        )
    lines.extend(("", "Relationship query first-relevant rank"))
    before_by_id = {query.query_id: query for query in reranked.queries}
    for after in expanded.queries:
        if after.category != "relationship":
            continue
        before = before_by_id[after.query_id]
        lines.append(_relationship_change(before, after))
    return "\n".join(lines)


def _relationship_change(before: QueryEvaluation, after: QueryEvaluation) -> str:
    if before.first_relevant_rank == after.first_relevant_rank:
        outcome = "unchanged"
    elif _rank_value(after.first_relevant_rank) < _rank_value(
        before.first_relevant_rank
    ):
        outcome = "helped"
    else:
        outcome = "regressed"
    return (
        f"{before.query_id}: {before.first_relevant_rank} -> "
        f"{after.first_relevant_rank} ({outcome})"
    )


def _rank_value(rank: int | None) -> float:
    return float("inf") if rank is None else float(rank)


def main() -> None:
    dense, rrf, reranked, expanded = run_comparison()
    print(
        format_report(reranked).replace(
            "BM25 Retrieval Baseline", "RRF + Cross-Encoder"
        )
    )
    print()
    print(
        format_report(expanded).replace(
            "BM25 Retrieval Baseline", "RRF + Cross-Encoder + Relationships"
        )
    )
    print()
    print(format_comparison(dense, rrf, reranked, expanded))


if __name__ == "__main__":
    main()
