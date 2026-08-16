from pathlib import Path

from evaluation.end_to_end import (
    EvaluationQuestion,
    RateLimitedGenerator,
    SystemOutput,
    deterministic_scores,
    load_dataset,
    parse_judge_result,
    run_system,
)
from langchain_core.documents import Document
from src.retrieval.relationship_expansion import ExpandedSearchResult

FIXTURE = Path("evaluation/fixtures/title_block_bom_eval.json")


class CitingGenerator:
    def generate(self, prompt: str) -> str:
        assert "Question:" in prompt
        return "The implementation returns the configured result [C1]."


class RateLimitOnceGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        from src.generation.groq import GenerationBackendError

        self.calls += 1
        if self.calls == 1:
            raise GenerationBackendError("Groq rate limit exceeded")
        return prompt


def result(source: str = "src/pipeline.py") -> ExpandedSearchResult:
    document = Document(
        page_content="class DrawingPipeline:\n    def run(self): ...",
        metadata={
            "source": source,
            "chunk_index": 0,
            "start_line": 10,
            "end_line": 11,
        },
    )
    return ExpandedSearchResult(
        document=document,
        origin="retrieved",
        original_rank=1,
        rerank_score=0.9,
        first_stage_score=0.8,
        first_stage_rank=1,
    )


def test_curated_dataset_is_frozen_and_has_required_category_balance() -> None:
    dataset = load_dataset(FIXTURE)

    assert dataset.expected_commit_sha == (
        "de4a29e4d453d0695b1a2cf5d76f8285032bea70"
    )
    assert len(dataset.questions) == 18
    counts = {
        category: sum(item.category == category for item in dataset.questions)
        for category in {
            "simple",
            "single-file",
            "cross-file",
            "architecture",
            "unanswerable",
        }
    }
    assert counts == {
        "simple": 4,
        "single-file": 5,
        "cross-file": 5,
        "architecture": 2,
        "unanswerable": 2,
    }
    assert all(item.expected_answer_points for item in dataset.questions)


def test_run_system_preserves_real_chunk_as_displayed_evidence() -> None:
    output = run_system(
        "Where is the pipeline?",
        lambda query, k: [result()],
        CitingGenerator(),
    )

    assert output.failure_reason is None
    assert output.citations == ("C1",)
    assert output.supporting_files == ("src/pipeline.py",)
    assert output.evidence[0].snippet == (
        "class DrawingPipeline:\n    def run(self): ..."
    )
    assert output.evidence[0].start_line == 10
    assert output.context_identifiers == ("src/pipeline.py::0",)


def test_file_overlap_scoring_is_deterministic() -> None:
    question = EvaluationQuestion(
        id="q1",
        question="How?",
        category="cross-file",
        difficulty="hard",
        answerable=True,
        expected_answer_points=("one",),
        expected_supporting_files=("a.py", "b.py"),
    )
    output = SystemOutput(
        answer="answer",
        citations=("C1", "C2"),
        supporting_files=("a.py", "extra.py"),
        evidence=(),
        retrieved_chunks=(),
        context_identifiers=(),
        retrieval_seconds=0.1,
        generation_seconds=0.2,
        total_seconds=0.3,
    )

    assert deterministic_scores(question, output) == {
        "citation_precision": 0.5,
        "citation_recall": 0.5,
    }


def test_judge_parser_validates_structured_scores_inside_code_fence() -> None:
    raw = '''```json
{"dense_only":{"correctness":2,"groundedness":3,"supporting_code_accuracy":2,"unanswerable_correct":null,"material_hallucination":false,"failure_stage":"generation","rationale":"partial"},"production":{"correctness":4,"groundedness":4,"supporting_code_accuracy":4,"unanswerable_correct":null,"material_hallucination":false,"failure_stage":null,"rationale":"complete"}}
```'''

    judged = parse_judge_result(raw)

    assert judged.dense_only.correctness == 2
    assert judged.production.correctness == 4
    assert judged.production.failure_stage is None


def test_rate_limit_wrapper_retries_without_changing_prompt() -> None:
    backend = RateLimitOnceGenerator()
    sleeps: list[float] = []
    generator = RateLimitedGenerator(
        backend,
        minimum_interval_seconds=0,
        rate_limit_retry_seconds=65,
        max_rate_limit_retries=1,
        sleep=sleeps.append,
    )

    assert generator.generate("safe prompt") == "safe prompt"
    assert backend.calls == 2
    assert sleeps == [65]
