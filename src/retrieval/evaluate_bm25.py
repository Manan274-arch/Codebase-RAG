"""Run the committed BM25 retrieval regression benchmark."""

from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_source_documents
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.evaluation import (
    RetrievalEvaluationResult,
    evaluate_retriever,
    load_evaluation_examples,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_REPOSITORY = (
    _PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_repo"
)
DEFAULT_BENCHMARK = _PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval.json"


def run_benchmark(
    repository: Path = DEFAULT_FIXTURE_REPOSITORY,
    benchmark: Path = DEFAULT_BENCHMARK,
) -> RetrievalEvaluationResult:
    """Build the fixture corpus and evaluate the raw-source BM25 baseline."""
    documents = load_source_documents(repository)
    chunks = chunk_documents(documents)
    examples = load_evaluation_examples(benchmark)
    return evaluate_retriever(BM25Retriever(chunks), chunks, examples)


def format_report(result: RetrievalEvaluationResult) -> str:
    """Format aggregate and category metrics as a compact deterministic table."""
    lines = ["BM25 Retrieval Baseline", f"Queries: {result.query_count}", ""]
    header = "Metric   " + "".join(f"@{k:<7}" for k in result.ks)
    lines.append(header.rstrip())
    rows = (
        ("HitRate", "hit_rate"),
        ("Recall", "recall"),
        ("MRR", "mrr"),
        ("nDCG", "ndcg"),
    )
    for label, field in rows:
        values = "".join(
            f"{getattr(result.metrics[k], field):<8.4f}" for k in result.ks
        )
        lines.append(f"{label:<9}{values}".rstrip())
    for category, metrics in result.category_metrics.items():
        lines.extend(("", f"Category: {category}"))
        for label, field in rows:
            values = "".join(f"{getattr(metrics[k], field):<8.4f}" for k in result.ks)
            lines.append(f"{label:<9}{values}".rstrip())
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":
    main()
