"""Ask one or more live questions about a public Git repository."""

import argparse
import sys

from src.codebase_rag import CodebaseRAG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="public HTTPS Git repository URL")
    parser.add_argument("--commit", help="optional full commit SHA")
    parser.add_argument(
        "--question",
        action="append",
        required=True,
        help="question to ask; repeat for multiple questions",
    )
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with CodebaseRAG.from_repository_url(
        arguments.repo,
        commit=arguments.commit,
    ) as rag:
        print(f"Repository URL: {rag.repository.repository_url}")
        print(f"Resolved commit SHA: {rag.repository.commit_sha}")
        print(f"Local checkout: {rag.repository.checkout_path}")
        print(f"Discovered source files: {rag.source_file_count}")
        print(f"Resulting chunks: {rag.chunk_count}")
        print("BM25 index status: ready")
        dense_status = "reused" if rag.index_result.reused else "built"
        print(f"Dense/Qdrant index status: {dense_status}")

        for question in arguments.question:
            result = rag.ask(question)
            print(f"\nQuestion: {question}")
            print("Generated answer:")
            print(result.answer)
            citations = ", ".join(result.citation_ids) or "<none>"
            print(f"Validated citation IDs: {citations}")
            print("Citation evidence:")
            for item in result.evidence:
                location = item.source
                if item.start_line is not None and item.end_line is not None:
                    location = f"{location}:{item.start_line}-{item.end_line}"
                print(f"[{item.citation_id}] {location}")
                print(item.page_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
