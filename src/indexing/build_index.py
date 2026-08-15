"""Build or validate the persistent local Qdrant code index."""

import argparse  # noqa: I001
from pathlib import Path

from src.retrieval import _torch_only  # noqa: F401
from src.config import QdrantSettings
from src.indexing.embeddings import SentenceTransformerBackend
from src.indexing.qdrant_index import QdrantCodeIndex
from src.pipeline.corpus import build_enriched_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="explicitly replace an existing stale or matching collection",
    )
    arguments = parser.parse_args()
    settings = QdrantSettings.from_environment()
    corpus = build_enriched_corpus(arguments.repository)
    index = QdrantCodeIndex(settings)
    try:
        result = index.ensure(
            corpus, SentenceTransformerBackend(), rebuild=arguments.rebuild
        )
    finally:
        index.close()
    action = "reused" if result.reused else "built"
    print(
        f"{action} {settings.collection_name!r}: {result.point_count} points; "
        f"fingerprint={result.fingerprint}"
    )


if __name__ == "__main__":
    main()
