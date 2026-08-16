"""Run production-versus-dense-only live RAG evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.end_to_end import load_dataset, run_evaluation

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "title_block_bom_eval.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="public HTTPS Git repository URL")
    parser.add_argument("--commit", help="optional full commit SHA")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/latest"),
    )
    parser.add_argument("--minimum-api-interval", type=float, default=8.0)
    parser.add_argument("--rate-limit-retry-seconds", type=float, default=65.0)
    parser.add_argument("--max-rate-limit-retries", type=int, default=3)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="rerun only outputs previously saved with a failure_reason",
    )
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dataset = load_dataset(arguments.dataset)
    result = run_evaluation(
        repository_url=arguments.repo,
        commit=arguments.commit,
        dataset=dataset,
        output_directory=arguments.output,
        minimum_interval_seconds=arguments.minimum_api_interval,
        rate_limit_retry_seconds=arguments.rate_limit_retry_seconds,
        max_rate_limit_retries=arguments.max_rate_limit_retries,
        retry_failed_outputs=arguments.retry_failures,
        progress=lambda position, total, question_id: print(
            f"Completed {position}/{total}: {question_id}", flush=True
        ),
    )
    print(f"Repository: {result['repository_url']}")
    print(f"Resolved commit: {result['resolved_commit_sha']}")
    print(f"Results: {arguments.output / 'results.json'}")
    print(f"Report: {arguments.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
