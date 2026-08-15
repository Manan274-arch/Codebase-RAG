import subprocess
from pathlib import Path

import pytest
from src.ingestion.acquisition import (
    RepositoryAcquisitionError,
    acquire_repository,
)


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def commit_file(repository: Path, content: str, message: str) -> str:
    (repository / "app.py").write_text(content, encoding="utf-8")
    git(repository, "add", "app.py")
    git(repository, "commit", "-m", message)
    return git(repository, "rev-parse", "HEAD")


def local_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "remote"
    repository.mkdir()
    git(repository, "init", "--initial-branch=main")
    git(repository, "config", "user.name", "Test User")
    git(repository, "config", "user.email", "test@example.invalid")
    first = commit_file(repository, "value = 1\n", "first")
    second = commit_file(repository, "value = 2\n", "second")
    return repository, first, second


def test_acquires_default_head_and_reuses_clean_checkout(tmp_path: Path) -> None:
    repository, _, head = local_repository(tmp_path)
    cache = tmp_path / "cache"

    first = acquire_repository(str(repository), cache_dir=cache)
    second = acquire_repository(str(repository), cache_dir=cache)

    assert first.commit_sha == head
    assert (first.checkout_path / "app.py").read_text(encoding="utf-8") == "value = 2\n"
    assert first.reused_checkout is False
    assert second.checkout_path == first.checkout_path
    assert second.reused_checkout is True
    assert git(first.checkout_path, "status", "--porcelain") == ""


def test_requested_commit_gets_distinct_immutable_checkout(tmp_path: Path) -> None:
    repository, first_sha, second_sha = local_repository(tmp_path)
    cache = tmp_path / "cache"

    old = acquire_repository(str(repository), first_sha, cache_dir=cache)
    current = acquire_repository(str(repository), second_sha, cache_dir=cache)

    assert old.commit_sha == first_sha
    assert current.commit_sha == second_sha
    assert old.checkout_path != current.checkout_path
    assert (old.checkout_path / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (current.checkout_path / "app.py").read_text(
        encoding="utf-8"
    ) == "value = 2\n"


def test_dirty_cached_checkout_is_recreated(tmp_path: Path) -> None:
    repository, _, _ = local_repository(tmp_path)
    cache = tmp_path / "cache"
    acquired = acquire_repository(str(repository), cache_dir=cache)
    (acquired.checkout_path / "app.py").write_text("modified\n", encoding="utf-8")

    repaired = acquire_repository(str(repository), cache_dir=cache)

    assert repaired.reused_checkout is True
    assert (repaired.checkout_path / "app.py").read_text(
        encoding="utf-8"
    ) == "value = 2\n"


def test_invalid_url_and_commit_fail_before_git_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        acquire_repository("ssh://example.com/repo.git", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="full hexadecimal"):
        acquire_repository("https://example.com/repo.git", "main", cache_dir=tmp_path)


def test_unknown_full_commit_fails_clearly(tmp_path: Path) -> None:
    repository, _, _ = local_repository(tmp_path)

    with pytest.raises(RepositoryAcquisitionError):
        acquire_repository(str(repository), "0" * 40, cache_dir=tmp_path / "cache")
