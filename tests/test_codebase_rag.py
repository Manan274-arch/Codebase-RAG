import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from src.codebase_rag import CodebaseRAG, CodebaseRAGError
from src.config import QdrantSettings


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> npt.NDArray[np.float32]:
        assert normalize_embeddings is True
        self.calls.append(tuple(texts))
        return np.asarray(
            [[1.0, float(len(text) % 7 + 1)] for text in texts],
            dtype=np.float32,
        )


class FakeScorer:
    def score(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> npt.NDArray[np.float32]:
        return np.asarray([1.0 for _ in pairs], dtype=np.float32)


class FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The answer function returns 42 [C1]."


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def repository(tmp_path: Path, *, source: bool = True) -> tuple[Path, str]:
    path = tmp_path / "remote"
    path.mkdir()
    git(path, "init", "--initial-branch=main")
    git(path, "config", "user.name", "Test User")
    git(path, "config", "user.email", "test@example.invalid")
    file = path / ("answer.py" if source else "README.md")
    file.write_text(
        "def answer() -> int:\n    return 42\n" if source else "No source\n",
        encoding="utf-8",
    )
    git(path, "add", file.name)
    git(path, "commit", "-m", "initial")
    return path, git(path, "rev-parse", "HEAD")


def test_prepares_once_and_reuses_pipeline_across_questions(tmp_path: Path) -> None:
    remote, commit = repository(tmp_path)
    encoder = FakeEncoder()
    generator = FakeGenerator()
    settings = QdrantSettings(
        path=tmp_path / "qdrant",
        collection_name="test_chunks",
        embedding_model="fake",
        vector_size=2,
    )

    with CodebaseRAG.from_repository_url(
        str(remote),
        cache_dir=tmp_path / "cache",
        qdrant_settings=settings,
        encoder=encoder,
        scorer=FakeScorer(),
        generator=generator,
    ) as rag:
        first = rag.ask("What does answer return?")
        second = rag.ask("Explain answer again")

        assert rag.repository.repository_url == str(remote)
        assert rag.repository.commit_sha == commit
        assert rag.source_file_count == 1
        assert rag.chunk_count == 1
        assert rag.index_result.reused is False
        assert first.answer == "The answer function returns 42 [C1]."
        assert first.citation_ids == ("C1",)
        assert first.evidence[0].source == "answer.py"
        assert second.citation_ids == ("C1",)

    assert len(encoder.calls) == 3
    assert len(generator.prompts) == 2


def test_repository_without_supported_sources_fails_clearly(tmp_path: Path) -> None:
    remote, _ = repository(tmp_path, source=False)

    with pytest.raises(CodebaseRAGError, match="no supported source files"):
        CodebaseRAG.from_repository_url(
            str(remote),
            cache_dir=tmp_path / "cache",
            generator=FakeGenerator(),
        )
