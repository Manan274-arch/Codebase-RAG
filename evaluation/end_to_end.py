"""Auditable end-to-end comparison of production and dense-only RAG."""

# ruff: noqa: I001 -- configure Transformers for PyTorch before model imports.

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval import _torch_only  # noqa: F401
from src.config import QdrantSettings
from src.generation.context import EvidenceItem, build_context
from src.generation.generator import TextGenerator, generate_answer
from src.generation.groq import GenerationBackendError, GroqTextGenerator
from src.indexing.embeddings import EmbeddingBackend, SentenceTransformerBackend
from src.indexing.qdrant_index import QdrantCodeIndex
from src.ingestion.acquisition import acquire_repository
from src.pipeline.corpus import build_enriched_corpus
from src.retrieval.contracts import canonical_chunk_id
from src.retrieval.pipeline import RetrievalPipeline, build_retrieval_pipeline
from src.retrieval.qdrant_dense import QdrantDenseRetriever
from src.retrieval.relationship_expansion import ExpandedSearchResult
from src.retrieval.reranker import PairScorer

QuestionCategory = Literal[
    "simple", "single-file", "cross-file", "architecture", "unanswerable"
]
Difficulty = Literal["easy", "medium", "hard"]
SystemName = Literal["dense_only", "production"]


class FrozenModel(BaseModel):
    """Strict immutable schema for benchmark and judge data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationQuestion(FrozenModel):
    id: str
    question: str
    category: QuestionCategory
    difficulty: Difficulty
    answerable: bool
    expected_answer_points: tuple[str, ...]
    expected_supporting_files: tuple[str, ...]
    expected_symbols: tuple[str, ...] = ()


class EvaluationDataset(FrozenModel):
    version: str
    repository_url: str
    expected_commit_sha: str | None = None
    questions: tuple[EvaluationQuestion, ...]


class JudgeSystemScore(FrozenModel):
    correctness: int = Field(ge=0, le=4)
    groundedness: int = Field(ge=0, le=4)
    supporting_code_accuracy: int = Field(ge=0, le=4)
    unanswerable_correct: bool | None
    material_hallucination: bool
    failure_stage: str | None
    rationale: str


class PairJudgeResult(FrozenModel):
    dense_only: JudgeSystemScore
    production: JudgeSystemScore


@dataclass(frozen=True, slots=True)
class RetrievedChunkRecord:
    evidence_id: str
    source: str
    chunk_index: int
    origin: str
    rerank_score: float | None
    first_stage_score: float | None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    citation_id: str
    evidence_id: str
    source: str
    chunk_index: int
    snippet: str
    start_line: int | None
    end_line: int | None
    origin: str


@dataclass(frozen=True, slots=True)
class SystemOutput:
    answer: str
    citations: tuple[str, ...]
    supporting_files: tuple[str, ...]
    evidence: tuple[EvidenceRecord, ...]
    retrieved_chunks: tuple[RetrievedChunkRecord, ...]
    context_identifiers: tuple[str, ...]
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    answer_correctness: float
    citation_precision: float
    citation_recall: float
    groundedness: float
    supporting_code_accuracy: float
    unanswerable_accuracy: float | None
    overall_success_rate: float


class RateLimitedGenerator:
    """Evaluation-only pacing/retry wrapper around unchanged generation settings."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        minimum_interval_seconds: float,
        rate_limit_retry_seconds: float,
        max_rate_limit_retries: int,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._generator = generator
        self._minimum_interval_seconds = minimum_interval_seconds
        self._rate_limit_retry_seconds = rate_limit_retry_seconds
        self._max_rate_limit_retries = max_rate_limit_retries
        self._sleep = sleep
        self._clock = clock
        self._last_call_finished: float | None = None

    def generate(self, prompt: str) -> str:
        retries = 0
        while True:
            self._pace()
            try:
                answer = self._generator.generate(prompt)
                self._last_call_finished = self._clock()
                return answer
            except GenerationBackendError as error:
                self._last_call_finished = self._clock()
                if "rate limit" not in str(error).lower():
                    raise
                if retries >= self._max_rate_limit_retries:
                    raise
                retries += 1
                self._sleep(self._rate_limit_retry_seconds)

    def _pace(self) -> None:
        if self._last_call_finished is None:
            return
        elapsed = self._clock() - self._last_call_finished
        remaining = self._minimum_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)


class DenseOnlyRetriever:
    """Evaluation-only adapter: exact dense results directly into context."""

    def __init__(self, dense: QdrantDenseRetriever) -> None:
        self._dense = dense

    def retrieve(self, query: str, k: int = 10) -> list[ExpandedSearchResult]:
        return [
            ExpandedSearchResult(
                document=result.document,
                origin="retrieved",
                original_rank=rank,
                rerank_score=None,
                first_stage_score=result.score,
                first_stage_rank=rank,
            )
            for rank, result in enumerate(self._dense.retrieve(query, k=k), start=1)
        ]


def load_dataset(path: Path) -> EvaluationDataset:
    """Load a human-auditable benchmark definition."""
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def run_evaluation(
    *,
    repository_url: str,
    commit: str | None,
    dataset: EvaluationDataset,
    output_directory: Path,
    generator: TextGenerator | None = None,
    judge_generator: TextGenerator | None = None,
    encoder: EmbeddingBackend | None = None,
    scorer: PairScorer | None = None,
    minimum_interval_seconds: float = 8.0,
    rate_limit_retry_seconds: float = 65.0,
    max_rate_limit_retries: int = 3,
    retry_failed_outputs: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Run both systems, judge after generation, and persist auditable artifacts."""
    repository = acquire_repository(repository_url, commit)
    if (
        dataset.expected_commit_sha is not None
        and dataset.expected_commit_sha != repository.commit_sha
    ):
        raise ValueError(
            "resolved commit does not match the dataset's expected_commit_sha"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    corpus = build_enriched_corpus(repository.checkout_path)
    settings = QdrantSettings.from_environment()
    settings = replace(
        settings,
        collection_name=(
            f"{settings.collection_name}_e2e_{repository.commit_sha[:16]}"
        ),
    )
    selected_encoder = encoder or SentenceTransformerBackend(settings.embedding_model)
    index = QdrantCodeIndex(settings)
    try:
        index_result = index.ensure(corpus, selected_encoder)
        production = build_retrieval_pipeline(
            corpus, index, selected_encoder, scorer=scorer
        )
        dense_only = DenseOnlyRetriever(production.dense)
        paced_generator = RateLimitedGenerator(
            generator or GroqTextGenerator(),
            minimum_interval_seconds=minimum_interval_seconds,
            rate_limit_retry_seconds=rate_limit_retry_seconds,
            max_rate_limit_retries=max_rate_limit_retries,
        )
        paced_judge = RateLimitedGenerator(
            judge_generator or GroqTextGenerator(max_tokens=2048),
            minimum_interval_seconds=minimum_interval_seconds,
            rate_limit_retry_seconds=rate_limit_retry_seconds,
            max_rate_limit_retries=max_rate_limit_retries,
        )

        partial_path = output_directory / "partial_results.json"
        pending_path = output_directory / "pending_question.json"
        existing_records = _load_partial_results(
            partial_path, dataset.questions
        )

        def checkpoint(records: Sequence[dict[str, object]]) -> None:
            partial_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        if retry_failed_outputs:
            existing_records = _repair_failed_records(
                existing_records,
                dataset.questions,
                dense_only,
                production,
                paced_generator,
                paced_judge,
                checkpoint,
                progress,
            )

        question_results = _run_questions(
            dataset.questions,
            dense_only,
            production,
            paced_generator,
            paced_judge,
            initial_records=existing_records,
            pending_path=pending_path,
            on_record=checkpoint,
            progress=progress,
        )
    finally:
        index.close()

    payload: dict[str, object] = {
        "benchmark_version": dataset.version,
        "repository_url": repository.repository_url,
        "resolved_commit_sha": repository.commit_sha,
        "source_file_count": len({doc.metadata["source"] for doc in corpus}),
        "chunk_count": len(corpus),
        "dense_index_reused": index_result.reused,
        "methodology": _methodology(),
        "questions": question_results,
    }
    aggregate = aggregate_results(question_results)
    payload["aggregate"] = aggregate
    (output_directory / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / "report.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    return payload


def _repair_failed_records(
    records: Sequence[dict[str, object]],
    questions: Sequence[EvaluationQuestion],
    dense_only: DenseOnlyRetriever,
    production: RetrievalPipeline,
    answer_generator: TextGenerator,
    judge_generator: TextGenerator,
    checkpoint: Callable[[Sequence[dict[str, object]]], None],
    progress: Callable[[int, int, str], None] | None,
) -> list[dict[str, object]]:
    repaired = list(records)
    for position, record in enumerate(repaired):
        question = questions[position]
        dense = _system_output_from_dict(record["dense_only"])
        final = _system_output_from_dict(record["production"])
        if dense.failure_reason is None and final.failure_reason is None:
            continue
        if dense.failure_reason is not None:
            dense = run_system(
                question.question, dense_only.retrieve, answer_generator
            )
        if final.failure_reason is not None:
            final = run_system(
                question.question, production.retrieve, answer_generator
            )
        judge_input = _judge_input(question, dense, final)
        raw_judge, judge = _judge_pair(judge_generator, judge_input)
        repaired[position] = _question_record(
            question, dense, final, judge_input, raw_judge, judge
        )
        checkpoint(repaired)
        if progress is not None:
            progress(position + 1, len(questions), f"{question.id} (repaired)")
    return repaired


def _run_questions(
    questions: Sequence[EvaluationQuestion],
    dense_only: DenseOnlyRetriever,
    production: RetrievalPipeline,
    answer_generator: TextGenerator,
    judge_generator: TextGenerator,
    *,
    initial_records: Sequence[dict[str, object]] = (),
    pending_path: Path | None = None,
    on_record: Callable[[Sequence[dict[str, object]]], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, object]]:
    records = list(initial_records)
    total = len(questions)
    remaining = questions[len(records) :]
    for position, question in enumerate(remaining, start=len(records) + 1):
        pending = _load_pending_output(pending_path, question.id)
        if pending is None:
            dense_output = run_system(
                question.question, dense_only.retrieve, answer_generator
            )
            production_output = run_system(
                question.question, production.retrieve, answer_generator
            )
            _save_pending_output(
                pending_path, question.id, dense_output, production_output
            )
        else:
            dense_output, production_output = pending
        judge_input = _judge_input(question, dense_output, production_output)
        raw_judge, judge = _judge_pair(judge_generator, judge_input)
        records.append(
            _question_record(
                question,
                dense_output,
                production_output,
                judge_input,
                raw_judge,
                judge,
            )
        )
        if pending_path is not None and pending_path.exists():
            pending_path.unlink()
        if on_record is not None:
            on_record(records)
        if progress is not None:
            progress(position, total, question.id)
    return records


def _question_record(
    question: EvaluationQuestion,
    dense: SystemOutput,
    production: SystemOutput,
    judge_input: dict[str, object],
    raw_judge: str,
    judge: PairJudgeResult,
) -> dict[str, object]:
    return {
        "ground_truth": question.model_dump(mode="json"),
        "dense_only": asdict(dense),
        "production": asdict(production),
        "judge_input": judge_input,
        "judge_raw_output": raw_judge,
        "judge": judge.model_dump(mode="json"),
        "deterministic": {
            "dense_only": deterministic_scores(question, dense),
            "production": deterministic_scores(question, production),
        },
    }


def run_system(
    question: str,
    retrieve: Callable[[str, int], Sequence[ExpandedSearchResult]],
    generator: TextGenerator,
) -> SystemOutput:
    """Run retrieval, context, generation, and evidence projection once."""
    started = time.perf_counter()
    retrieval_started = started
    try:
        retrieved = tuple(retrieve(question, 10))
        retrieval_seconds = time.perf_counter() - retrieval_started
        context = build_context(retrieved)
        generation_started = time.perf_counter()
        generated = generate_answer(question, context, generator)
        generation_seconds = time.perf_counter() - generation_started
        evidence_by_id = {item.citation_id: item for item in context.evidence}
        cited = tuple(evidence_by_id[item] for item in generated.citation_ids)
        return SystemOutput(
            answer=generated.answer,
            citations=generated.citation_ids,
            supporting_files=_ordered_unique(item.source for item in cited),
            evidence=tuple(_evidence_record(item) for item in cited),
            retrieved_chunks=tuple(_retrieved_record(item) for item in retrieved),
            context_identifiers=tuple(item.evidence_id for item in context.evidence),
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        return SystemOutput(
            answer="",
            citations=(),
            supporting_files=(),
            evidence=(),
            retrieved_chunks=(),
            context_identifiers=(),
            retrieval_seconds=0.0,
            generation_seconds=0.0,
            total_seconds=time.perf_counter() - started,
            failure_reason=f"{type(error).__name__}: {error}",
        )


def deterministic_scores(
    question: EvaluationQuestion, output: SystemOutput
) -> dict[str, float]:
    expected = set(question.expected_supporting_files)
    cited = set(output.supporting_files)
    precision = len(expected & cited) / len(cited) if cited else 0.0
    recall = len(expected & cited) / len(expected) if expected else 1.0
    return {"citation_precision": precision, "citation_recall": recall}


def parse_judge_result(raw: str) -> PairJudgeResult:
    """Validate the judge's strict JSON, tolerating only surrounding prose/fences."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON object")
    return PairJudgeResult.model_validate_json(raw[start : end + 1])


def _judge_pair(
    generator: TextGenerator, judge_input: dict[str, object]
) -> tuple[str, PairJudgeResult]:
    last_error: ValueError | None = None
    for _ in range(2):
        raw = generator.generate(_judge_prompt(judge_input))
        try:
            return raw, parse_judge_result(raw)
        except ValueError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def _load_partial_results(
    path: Path, questions: Sequence[EvaluationQuestion]
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("partial_results.json must contain a list")
    records = cast(list[dict[str, object]], raw)
    if len(records) > len(questions):
        raise ValueError("partial results contain too many questions")
    for position, record in enumerate(records):
        ground_truth = EvaluationQuestion.model_validate(record.get("ground_truth"))
        if ground_truth.id != questions[position].id:
            raise ValueError("partial results do not match the dataset question order")
    return records


def _save_pending_output(
    path: Path | None,
    question_id: str,
    dense: SystemOutput,
    production: SystemOutput,
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {
                "question_id": question_id,
                "dense_only": asdict(dense),
                "production": asdict(production),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_pending_output(
    path: Path | None, question_id: str
) -> tuple[SystemOutput, SystemOutput] | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("question_id") != question_id:
        raise ValueError("pending question does not match the next dataset question")
    return (
        _system_output_from_dict(raw.get("dense_only")),
        _system_output_from_dict(raw.get("production")),
    )


def _system_output_from_dict(value: object) -> SystemOutput:
    if not isinstance(value, dict):
        raise ValueError("pending system output must be an object")
    data = cast(dict[str, object], value)
    evidence_values = data.get("evidence")
    retrieved_values = data.get("retrieved_chunks")
    if not isinstance(evidence_values, list) or not isinstance(retrieved_values, list):
        raise ValueError("pending system output has invalid evidence")
    return SystemOutput(
        answer=cast(str, data["answer"]),
        citations=tuple(cast(list[str], data["citations"])),
        supporting_files=tuple(cast(list[str], data["supporting_files"])),
        evidence=tuple(
            _evidence_record_from_dict(item)
            for item in evidence_values
        ),
        retrieved_chunks=tuple(
            _retrieved_record_from_dict(item)
            for item in retrieved_values
        ),
        context_identifiers=tuple(
            cast(list[str], data["context_identifiers"])
        ),
        retrieval_seconds=float(cast(float, data["retrieval_seconds"])),
        generation_seconds=float(cast(float, data["generation_seconds"])),
        total_seconds=float(cast(float, data["total_seconds"])),
        failure_reason=cast(str | None, data.get("failure_reason")),
    )


def _evidence_record_from_dict(value: object) -> EvidenceRecord:
    if not isinstance(value, dict):
        raise ValueError("pending evidence entry must be an object")
    data = cast(dict[str, object], value)
    return EvidenceRecord(
        citation_id=cast(str, data["citation_id"]),
        evidence_id=cast(str, data["evidence_id"]),
        source=cast(str, data["source"]),
        chunk_index=cast(int, data["chunk_index"]),
        snippet=cast(str, data["snippet"]),
        start_line=cast(int | None, data["start_line"]),
        end_line=cast(int | None, data["end_line"]),
        origin=cast(str, data["origin"]),
    )


def _retrieved_record_from_dict(value: object) -> RetrievedChunkRecord:
    if not isinstance(value, dict):
        raise ValueError("pending retrieved entry must be an object")
    data = cast(dict[str, object], value)
    return RetrievedChunkRecord(
        evidence_id=cast(str, data["evidence_id"]),
        source=cast(str, data["source"]),
        chunk_index=cast(int, data["chunk_index"]),
        origin=cast(str, data["origin"]),
        rerank_score=cast(float | None, data["rerank_score"]),
        first_stage_score=cast(float | None, data["first_stage_score"]),
    )


def aggregate_results(records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate global and category metrics from detailed saved records."""
    aggregate: dict[str, object] = {}
    aggregate["overall"] = {
        system: asdict(_aggregate_system(records, cast(SystemName, system)))
        for system in ("dense_only", "production")
    }
    categories = (
        "simple",
        "single-file",
        "cross-file",
        "architecture",
        "unanswerable",
    )
    aggregate["by_category"] = {
        category: {
            system: asdict(
                _aggregate_system(
                    [
                        record
                        for record in records
                        if _record_category(record) == category
                    ],
                    cast(SystemName, system),
                )
            )
            for system in ("dense_only", "production")
        }
        for category in categories
    }
    return aggregate


def _aggregate_system(
    records: Sequence[dict[str, object]], system: SystemName
) -> SystemMetrics:
    if not records:
        return SystemMetrics(0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0)
    correctness: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    groundedness: list[float] = []
    code_accuracy: list[float] = []
    negative_accuracy: list[float] = []
    successes: list[float] = []
    for record in records:
        judge = PairJudgeResult.model_validate(record["judge"])
        score = getattr(judge, system)
        deterministic = cast(dict[str, dict[str, float]], record["deterministic"])[
            system
        ]
        question = EvaluationQuestion.model_validate(record["ground_truth"])
        correctness_value = score.correctness / 4
        groundedness_value = score.groundedness / 4
        code_value = score.supporting_code_accuracy / 4
        correctness.append(correctness_value)
        precision.append(deterministic["citation_precision"])
        recall.append(deterministic["citation_recall"])
        groundedness.append(groundedness_value)
        code_accuracy.append(code_value)
        if not question.answerable:
            negative_accuracy.append(1.0 if score.unanswerable_correct else 0.0)
        successes.append(
            1.0
            if _meets_success_rule(question, score, deterministic)
            else 0.0
        )
    return SystemMetrics(
        answer_correctness=_mean(correctness),
        citation_precision=_mean(precision),
        citation_recall=_mean(recall),
        groundedness=_mean(groundedness),
        supporting_code_accuracy=_mean(code_accuracy),
        unanswerable_accuracy=(
            _mean(negative_accuracy) if negative_accuracy else None
        ),
        overall_success_rate=_mean(successes),
    )


def render_report(payload: dict[str, object]) -> str:
    """Render the measured comparison and category breakdown as Markdown."""
    aggregate = cast(dict[str, object], payload["aggregate"])
    overall = cast(
        dict[str, dict[str, float | None]], aggregate["overall"]
    )
    lines = [
        "# End-to-End RAG Comparison",
        "",
        f"- Repository: `{payload['repository_url']}`",
        f"- Resolved commit: `{payload['resolved_commit_sha']}`",
        f"- Benchmark: `{payload['benchmark_version']}`",
        f"- Questions: {len(cast(list[object], payload['questions']))}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Dense-only baseline | Final RAG | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in _METRIC_LABELS:
        dense = overall["dense_only"][key]
        production = overall["production"][key]
        if dense is None or production is None:
            lines.append(f"| {label} | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {label} | {dense:.1%} | {production:.1%} | "
            f"{production - dense:+.1%} |"
        )
    differences = [
        (
            label,
            cast(float, overall["production"][key])
            - cast(float, overall["dense_only"][key]),
        )
        for key, label in _METRIC_LABELS
        if overall["dense_only"][key] is not None
        and overall["production"][key] is not None
    ]
    wins = sorted(
        ((label, difference) for label, difference in differences if difference > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    losses = sorted(
        ((label, difference) for label, difference in differences if difference < 0),
        key=lambda item: item[1],
    )
    lines.extend(["", "## Observed comparison", ""])
    lines.append(
        "Final RAG improvements: "
        + (
            ", ".join(f"{label} {difference:+.1%}" for label, difference in wins)
            if wins
            else "none measured"
        )
        + "."
    )
    lines.append(
        "Final RAG regressions: "
        + (
            ", ".join(f"{label} {difference:+.1%}" for label, difference in losses)
            if losses
            else "none measured"
        )
        + "."
    )
    lines.extend(["", "## Category metrics", ""])
    by_category = cast(
        dict[str, dict[str, dict[str, float | None]]], aggregate["by_category"]
    )
    for category, systems in by_category.items():
        lines.extend(
            [
                f"### {category}",
                "",
                "| Metric | Dense-only | Final RAG |",
                "| --- | ---: | ---: |",
            ]
        )
        for key, label in _METRIC_LABELS:
            dense = systems["dense_only"][key]
            production = systems["production"][key]
            if dense is None or production is None:
                lines.append(f"| {label} | N/A | N/A |")
            else:
                lines.append(
                    f"| {label} | {dense:.1%} | {production:.1%} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Failure analysis",
            "",
            "A failure means the strict overall-success rule was not met; it does "
            "not necessarily mean the answer was factually wrong.",
            "",
            "| Question | System | Likely stage | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    records = cast(list[dict[str, object]], payload["questions"])
    for record in records:
        question = EvaluationQuestion.model_validate(record["ground_truth"])
        judge = PairJudgeResult.model_validate(record["judge"])
        deterministic = cast(
            dict[str, dict[str, float]], record["deterministic"]
        )
        for system in ("dense_only", "production"):
            system_name: SystemName = system
            score = getattr(judge, system_name)
            file_scores = deterministic[system_name]
            if _meets_success_rule(question, score, file_scores):
                continue
            stage = _likely_failure_stage(score, file_scores)
            rationale = score.rationale.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {question.id} | {system_name} | {stage} | {rationale} |"
            )
    lines.append("")
    lines.extend(
        [
            "## Methodology",
            "",
            str(payload["methodology"]),
            "",
            "Detailed answers, citations, snippets, retrieval/context identifiers, "
            "latencies, judge inputs, raw judge outputs, and failure classifications "
            "are retained in `results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _meets_success_rule(
    question: EvaluationQuestion,
    score: JudgeSystemScore,
    deterministic: dict[str, float],
) -> bool:
    return (
        not score.material_hallucination
        and score.correctness / 4 >= 0.75
        and deterministic["citation_precision"] >= 0.75
        and deterministic["citation_recall"] >= 0.5
        and score.groundedness / 4 >= 0.75
        and score.supporting_code_accuracy / 4 >= 0.75
        and (question.answerable or bool(score.unanswerable_correct))
    )


def _likely_failure_stage(
    score: JudgeSystemScore, deterministic: dict[str, float]
) -> str:
    if score.failure_stage is not None:
        return score.failure_stage
    if (
        deterministic["citation_precision"] < 0.75
        or deterministic["citation_recall"] < 0.5
    ):
        return "citation mapping"
    if (
        score.correctness / 4 < 0.75
        or score.groundedness / 4 < 0.75
        or score.supporting_code_accuracy / 4 < 0.75
        or score.material_hallucination
    ):
        return "generation"
    return "unanswerable handling"


_METRIC_LABELS = (
    ("answer_correctness", "Answer correctness"),
    ("citation_precision", "Citation precision"),
    ("citation_recall", "Citation recall"),
    ("groundedness", "Groundedness"),
    ("supporting_code_accuracy", "Supporting-code accuracy"),
    ("unanswerable_accuracy", "Unanswerable accuracy"),
    ("overall_success_rate", "Overall success rate"),
)


def _judge_input(
    question: EvaluationQuestion,
    dense: SystemOutput,
    production: SystemOutput,
) -> dict[str, object]:
    return {
        "ground_truth": question.model_dump(mode="json"),
        "dense_only": _judge_system_input(dense),
        "production": _judge_system_input(production),
    }


def _judge_system_input(output: SystemOutput) -> dict[str, object]:
    """Keep judge requests bounded while full evidence stays in saved results."""
    return {
        "answer": output.answer,
        "citations": output.citations,
        "supporting_files": output.supporting_files,
        "evidence": [
            {
                "citation_id": item.citation_id,
                "source": item.source,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "snippet_preview": item.snippet[:600],
            }
            for item in output.evidence
        ],
        "failure_reason": output.failure_reason,
    }


def _judge_prompt(judge_input: dict[str, object]) -> str:
    return (
        "You are a strict, impartial evaluator of two codebase-Q&A outputs. The "
        "answers have already been generated; ground truth was never shown to either "
        "system. Score each system independently from the supplied expected answer "
        "points and repository snippets. Correctness, groundedness, and supporting "
        "code accuracy use integers 0-4: 0 none/contradicted, 1 major defects, 2 "
        "partial, 3 substantially correct with minor omissions, 4 complete and "
        "correct. material_hallucination is true for any important unsupported or "
        "contradicted claim. For an unanswerable question, unanswerable_correct is "
        "true only if the answer clearly refuses or qualifies the premise without "
        "inventing an implementation; otherwise use null. failure_stage must be null "
        "or one concise label from ingestion/chunking, dense/BM25 retrieval, candidate "
        "union, reranking, relationship expansion, context construction, generation, "
        "citation mapping, frontend snippet presentation. Return JSON only with keys "
        "dense_only and production, each containing correctness, groundedness, "
        "supporting_code_accuracy, unanswerable_correct, material_hallucination, "
        "failure_stage, rationale.\n\n"
        "Keep each rationale under 20 words and emit compact JSON with no Markdown.\n\n"
        + json.dumps(judge_input, ensure_ascii=False)
    )


def _retrieved_record(result: ExpandedSearchResult) -> RetrievedChunkRecord:
    source = result.document.metadata["source"]
    chunk_index = result.document.metadata["chunk_index"]
    assert isinstance(source, str)
    assert isinstance(chunk_index, int)
    return RetrievedChunkRecord(
        evidence_id=canonical_chunk_id(result.document),
        source=source,
        chunk_index=chunk_index,
        origin=result.origin,
        rerank_score=result.rerank_score,
        first_stage_score=result.first_stage_score,
    )


def _evidence_record(item: EvidenceItem) -> EvidenceRecord:
    return EvidenceRecord(
        citation_id=item.citation_id,
        evidence_id=item.evidence_id,
        source=item.source,
        chunk_index=item.chunk_index,
        snippet=item.page_content,
        start_line=item.start_line,
        end_line=item.end_line,
        origin=item.origin,
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _record_category(record: dict[str, object]) -> str:
    question = EvaluationQuestion.model_validate(record["ground_truth"])
    return question.category


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _methodology() -> str:
    return (
        "Citation precision/recall are deterministic exact repository-relative file "
        "overlap against manually curated supporting files. A separate post-generation "
        "Groq rubric judge scores correctness, groundedness, supporting-code accuracy, "
        "and hallucination from 0-4; its complete bounded inputs and raw outputs are "
        "saved, while results retain every full displayed snippet. "
        "The two answer systems use identical default 1,024-token generation settings; "
        "the post-generation structured judge uses a 2,048-token output allowance. "
        "Overall answerable success requires >=75% correctness, >=75% citation "
        "precision, >=50% citation recall, >=75% groundedness, >=75% supporting-code "
        "accuracy, and no material hallucination. Unanswerable success additionally "
        "requires a correct refusal/qualification while retaining those same "
        "citation, groundedness, and supporting-code thresholds."
    )
