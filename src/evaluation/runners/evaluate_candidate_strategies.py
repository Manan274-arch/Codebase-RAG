"""Controlled candidate-generation experiment on frozen Benchmark v2."""

from collections.abc import Sequence  # noqa: I001
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from src.retrieval import _torch_only  # noqa: F401
from src.ingestion.pipeline import build_enriched_corpus
from src.evaluation.benchmark_v2 import (
    BENCHMARK_V2_QUERIES,
    BENCHMARK_V2_REPOSITORY,
    linked_context_coverage,
    load_and_validate_benchmark_v2,
)
from src.retrieval.bm25 import BM25Retriever
from src.evaluation.candidate_diagnostics import count_union_provenance
from src.retrieval.candidate_union import CandidateUnionRetriever
from src.evaluation.brute_force_dense import DenseRetriever
from src.evaluation.metrics import (
    DocumentRetriever,
    QueryEvaluation,
    RetrievalEvaluationExample,
    RetrievalEvaluationResult,
    evaluate_retriever,
    subset_evaluation_result,
)
from src.evaluation.rrf import HybridRetriever
from src.retrieval.relationship_expansion import RelationshipExpander
from src.retrieval.reranker import (
    CrossEncoderReranker,
    SentenceTransformerCrossEncoder,
)

RAW_KS = (10, 20, 50)
CANDIDATE_DIAGNOSTIC_KS = (3, *RAW_KS)
FINAL_KS = (1, 3, 5, 10)
UNION_BRANCH_DEPTH = 25
FINAL_CANDIDATE_DEPTH = 50


class CachedRetriever:
    """Cache largest per-query results so expansion reuses cross-encoder work."""

    def __init__(self, retriever: DocumentRetriever) -> None:
        self._retriever = retriever
        self._cache: dict[str, tuple[int, list[Any]]] = {}

    def retrieve(self, query: str, k: int = 10) -> Sequence[Any]:
        cached = self._cache.get(query)
        if cached is None or cached[0] < k:
            results = list(self._retriever.retrieve(query, k=k))
            cached = (k, results)
            self._cache[query] = cached
        return cached[1][:k]


@dataclass(frozen=True, slots=True)
class StrategyExperiment:
    corpus: tuple[Document, ...]
    examples: tuple[RetrievalEvaluationExample, ...]
    candidates: dict[str, RetrievalEvaluationResult]
    reranked: dict[str, RetrievalEvaluationResult]
    expanded: dict[str, RetrievalEvaluationResult]
    runtimes: dict[str, float]
    provenance: dict[str, tuple[int, int, int]]


def run_experiment() -> StrategyExperiment:
    """Run fixed 50-candidate strategies once across both stored splits."""
    corpus = tuple(build_enriched_corpus(BENCHMARK_V2_REPOSITORY))
    examples = load_and_validate_benchmark_v2(corpus, BENCHMARK_V2_QUERIES)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    rrf = HybridRetriever(bm25, dense, candidate_depth=FINAL_CANDIDATE_DEPTH)
    union = CandidateUnionRetriever(
        bm25,
        dense,
        bm25_depth=UNION_BRANCH_DEPTH,
        dense_depth=UNION_BRANCH_DEPTH,
    )
    candidate_retrievers: dict[str, DocumentRetriever] = {
        "BM25": bm25,
        "Dense": dense,
        "RRF": rrf,
        "Union": union,
    }
    candidates = {
        name: evaluate_retriever(
            retriever, corpus, examples, ks=CANDIDATE_DIAGNOSTIC_KS
        )
        for name, retriever in candidate_retrievers.items()
    }
    provenance: dict[str, tuple[int, int, int]] = {}
    for split in ("dev", "test"):
        split_examples = tuple(e for e in examples if e.split == split)
        counts = count_union_provenance(union, split_examples, k=50)
        provenance[split] = (counts.bm25_only, counts.dense_only, counts.both)

    scorer = SentenceTransformerCrossEncoder()
    final_inputs: dict[str, DocumentRetriever] = {
        "Dense": dense,
        "RRF": rrf,
        "Union": union,
    }
    reranked: dict[str, RetrievalEvaluationResult] = {}
    expanded: dict[str, RetrievalEvaluationResult] = {}
    runtimes: dict[str, float] = {}
    for name, candidate_retriever in final_inputs.items():
        started = perf_counter()
        cached = CachedRetriever(
            CrossEncoderReranker(
                candidate_retriever,
                scorer=scorer,
                candidate_depth=FINAL_CANDIDATE_DEPTH,
            )
        )
        reranked[name] = evaluate_retriever(
            cached, corpus, examples, ks=(*FINAL_KS, FINAL_CANDIDATE_DEPTH)
        )
        expanded[name] = evaluate_retriever(
            RelationshipExpander(cached, corpus), corpus, examples, ks=FINAL_KS
        )
        runtimes[name] = perf_counter() - started
    return StrategyExperiment(
        corpus, examples, candidates, reranked, expanded, runtimes, provenance
    )


def format_experiment(run: StrategyExperiment) -> str:
    """Format candidate diagnostics, final metrics, categories, and differences."""
    lines = [
        "Candidate Strategy Experiment",
        "Configuration: Dense 50; RRF 50 from 50+50; "
        "Union <=50 from BM25 25 + Dense 25",
    ]
    for split in ("dev", "test"):
        ids = frozenset(e.query_id for e in run.examples if e.split == split)
        lines.extend(
            ("", f"{split.upper()} candidate recall/hit", _candidate_table(run, ids))
        )
        bm25_only, dense_only, both = run.provenance[split]
        lines.append(
            f"Union occurrences @50: bm25_only={bm25_only} "
            f"dense_only={dense_only} both={both}"
        )
        lines.extend(
            ("", f"{split.upper()} expanded final metrics", _final_table(run, ids))
        )
        if split == "test":
            lines.extend(("", "TEST expanded categories", _category_table(run, ids)))
            lines.extend(("", "TEST linked coverage@5", _coverage_table(run, ids)))
    lines.extend(("", "Pipeline runtime seconds"))
    lines.extend(f"{name}: {seconds:.2f}" for name, seconds in run.runtimes.items())
    lines.extend(("", "Representative strategy differences"))
    lines.extend(_difference_lines(run))
    return "\n".join(lines)


def _candidate_table(run: StrategyExperiment, ids: frozenset[str]) -> str:
    lines = ["Stage   Recall@10/20/50          Hit@10/20/50"]
    for name, full in run.candidates.items():
        result = subset_evaluation_result(full, ids)
        recall = "/".join(f"{result.metrics[k].recall:.4f}" for k in RAW_KS)
        hit = "/".join(f"{result.metrics[k].hit_rate:.4f}" for k in RAW_KS)
        lines.append(f"{name:<8}{recall:<25}{hit}")
    return "\n".join(lines)


def _final_table(run: StrategyExperiment, ids: frozenset[str]) -> str:
    lines = ["Stage   Hit@1/3/5/10  Recall@1/3/5/10  MRR@1/3/5/10  nDCG@1/3/5/10"]
    for name, full in run.expanded.items():
        result = subset_evaluation_result(full, ids)
        groups = [
            "/".join(f"{getattr(result.metrics[k], field):.4f}" for k in FINAL_KS)
            for field in ("hit_rate", "recall", "mrr", "ndcg")
        ]
        lines.append(f"{name:<8}" + "  ".join(groups))
    return "\n".join(lines)


def _category_table(run: StrategyExperiment, ids: frozenset[str]) -> str:
    lines = ["Stage   Category      Hit@1   Hit@3   MRR@5   nDCG@5"]
    for name, full in run.expanded.items():
        result = subset_evaluation_result(full, ids)
        for category, metrics in result.category_metrics.items():
            lines.append(
                f"{name:<8}{category:<14}{metrics[1].hit_rate:<8.4f}"
                f"{metrics[3].hit_rate:<8.4f}{metrics[5].mrr:<8.4f}"
                f"{metrics[5].ndcg:.4f}"
            )
    return "\n".join(lines)


def _coverage_table(run: StrategyExperiment, ids: frozenset[str]) -> str:
    examples = tuple(e for e in run.examples if e.query_id in ids)
    lines: list[str] = []
    for name, full in run.expanded.items():
        coverage, eligible = linked_context_coverage(
            subset_evaluation_result(full, ids), examples, run.corpus, k=5
        )
        lines.append(f"{name}: {coverage:.4f} (eligible={eligible})")
    return "\n".join(lines)


def _difference_lines(run: StrategyExperiment) -> list[str]:
    test_ids = frozenset(e.query_id for e in run.examples if e.split == "test")
    candidate = {
        name: {
            q.query_id: q for q in subset_evaluation_result(result, test_ids).queries
        }
        for name, result in run.candidates.items()
    }
    final = {
        name: {
            q.query_id: q for q in subset_evaluation_result(result, test_ids).queries
        }
        for name, result in run.expanded.items()
    }
    examples = {e.query_id: e for e in run.examples if e.query_id in test_ids}
    conditions = (
        ("dense_ok_rrf_fail", "Dense", "RRF"),
        ("rrf_ok_dense_fail", "RRF", "Dense"),
        ("union_ok_dense_fail", "Union", "Dense"),
        ("dense_ok_union_fail", "Dense", "Union"),
    )
    lines: list[str] = []
    for label, success, failure in conditions:
        matches = [
            query_id
            for query_id in sorted(test_ids)
            if final[success][query_id].metrics[3].hit_rate
            and not final[failure][query_id].metrics[3].hit_rate
        ][:3]
        for query_id in matches:
            lines.append(_diagnostic(label, query_id, examples, candidate, final))
        if not matches:
            lines.append(f"{label}|no @3 cases")
    ordering_conditions = (
        ("dense_better_ordering", "Dense", "RRF"),
        ("rrf_better_ordering", "RRF", "Dense"),
        ("union_better_ordering", "Union", "Dense"),
        ("dense_better_than_union", "Dense", "Union"),
    )
    for label, better, worse in ordering_conditions:
        matches = sorted(
            test_ids,
            key=lambda query_id: (
                final[better][query_id].metrics[5].ndcg
                - final[worse][query_id].metrics[5].ndcg,
                query_id,
            ),
            reverse=True,
        )
        matches = [
            query_id
            for query_id in matches
            if final[better][query_id].metrics[5].ndcg
            > final[worse][query_id].metrics[5].ndcg
        ][:2]
        for query_id in matches:
            lines.append(_diagnostic(label, query_id, examples, candidate, final))
        if not matches:
            lines.append(f"{label}|no nDCG@5 cases")
    for strategy in ("Dense", "RRF", "Union"):
        matches = [
            query_id
            for query_id in sorted(test_ids)
            if candidate[strategy][query_id].metrics[3].hit_rate
            and not final[strategy][query_id].metrics[3].hit_rate
        ][:2]
        for query_id in matches:
            lines.append(
                _diagnostic(
                    f"cross_regression_{strategy.lower()}",
                    query_id,
                    examples,
                    candidate,
                    final,
                )
            )
    return lines or ["No requested @3 success/failure differences found."]


def _diagnostic(
    label: str,
    query_id: str,
    examples: dict[str, RetrievalEvaluationExample],
    candidates: dict[str, dict[str, QueryEvaluation]],
    finals: dict[str, dict[str, QueryEvaluation]],
) -> str:
    example = examples[query_id]
    expected = sorted(example.relevant_chunk_ids)
    positions = {
        name: _first_position(
            results[query_id].retrieved_chunk_ids, example.relevant_chunk_ids
        )
        for name, results in candidates.items()
    }
    ranks = {
        name: _first_position(
            results[query_id].retrieved_chunk_ids, example.relevant_chunk_ids
        )
        for name, results in finals.items()
    }
    return (
        f"{label}|{query_id}|{example.category}|query={example.query}"
        f"|expected={','.join(expected)}|candidate_positions={positions}|final_ranks={ranks}"
    )


def _first_position(ranked: Sequence[str], relevant: frozenset[str]) -> int | None:
    return next((rank for rank, item in enumerate(ranked, 1) if item in relevant), None)


def main() -> None:
    print(format_experiment(run_experiment()))


if __name__ == "__main__":
    main()
