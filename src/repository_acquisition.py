"""Safe cached acquisition of public Git repositories at immutable commits."""

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_REPOSITORY_CACHE = Path(".cache") / "codebase-rag" / "repositories"
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")


class RepositoryAcquisitionError(RuntimeError):
    """Raised when a repository cannot be fetched at a trustworthy commit."""


@dataclass(frozen=True, slots=True)
class AcquiredRepository:
    """A clean detached checkout with stable remote identity."""

    repository_url: str
    commit_sha: str
    checkout_path: Path
    reused_checkout: bool


def acquire_repository(
    repository_url: str,
    commit: str | None = None,
    *,
    cache_dir: Path = DEFAULT_REPOSITORY_CACHE,
) -> AcquiredRepository:
    """Fetch ``repository_url`` and return a clean checkout pinned to one SHA."""
    normalized_url = _validate_repository_url(repository_url)
    requested_commit = _validate_commit(commit)
    cache_dir = cache_dir.resolve()
    repository_key = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:20]
    repository_cache = cache_dir / repository_key
    mirror = repository_cache / "mirror.git"
    hooks = repository_cache / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)

    if mirror.exists():
        if not _is_bare_repository(mirror, hooks):
            raise RepositoryAcquisitionError(
                f"cached mirror is invalid: {mirror}"
            )
        remote_url = _git(
            [f"--git-dir={mirror}", "remote", "get-url", "origin"], hooks=hooks
        )
        if remote_url != normalized_url:
            raise RepositoryAcquisitionError(
                "cached mirror remote does not match the requested repository"
            )
    else:
        _clone_mirror(normalized_url, mirror, hooks)

    _git([f"--git-dir={mirror}", "fetch", "--prune", "origin"], hooks=hooks)
    resolved_commit = _resolve_commit(mirror, requested_commit, hooks)
    checkout = repository_cache / "checkouts" / resolved_commit
    reused = _reuse_clean_checkout(checkout, resolved_commit, hooks)
    if not reused:
        _create_checkout(mirror, checkout, resolved_commit, hooks)

    return AcquiredRepository(
        repository_url=repository_url,
        commit_sha=resolved_commit,
        checkout_path=checkout,
        reused_checkout=reused,
    )


def _validate_repository_url(repository_url: str) -> str:
    if not isinstance(repository_url, str) or not repository_url.strip():
        raise ValueError("repository_url must be a non-empty string")
    value = repository_url.strip()
    if value.startswith("-") or "\n" in value or "\r" in value:
        raise ValueError("repository_url is invalid")
    local = Path(value)
    if local.exists():
        return str(local.resolve())
    parsed = urlparse(value)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("only public HTTPS Git repository URLs are supported")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("repository URLs must not contain credentials")
        return value
    raise ValueError("repository_url must be a public HTTPS URL")


def _validate_commit(commit: str | None) -> str | None:
    if commit is None:
        return None
    if not isinstance(commit, str) or _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("commit must be a full hexadecimal Git commit SHA")
    return commit.lower()


def _clone_mirror(repository_url: str, mirror: Path, hooks: Path) -> None:
    mirror.parent.mkdir(parents=True, exist_ok=True)
    temporary = mirror.with_name("mirror.git.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        _git(["clone", "--mirror", repository_url, str(temporary)], hooks=hooks)
        temporary.replace(mirror)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _resolve_commit(mirror: Path, commit: str | None, hooks: Path) -> str:
    if commit is None:
        _git([f"--git-dir={mirror}", "fetch", "origin", "HEAD"], hooks=hooks)
        return _git(
            [f"--git-dir={mirror}", "rev-parse", "--verify", "FETCH_HEAD^{commit}"],
            hooks=hooks,
        ).lower()
    try:
        resolved = _git(
            [f"--git-dir={mirror}", "rev-parse", "--verify", f"{commit}^{{commit}}"],
            hooks=hooks,
        ).lower()
    except RepositoryAcquisitionError:
        _git([f"--git-dir={mirror}", "fetch", "origin", commit], hooks=hooks)
        resolved = _git(
            [f"--git-dir={mirror}", "rev-parse", "--verify", "FETCH_HEAD^{commit}"],
            hooks=hooks,
        ).lower()
    if resolved != commit:
        raise RepositoryAcquisitionError(
            f"requested commit was not resolved exactly: {commit}"
        )
    return resolved


def _reuse_clean_checkout(checkout: Path, commit: str, hooks: Path) -> bool:
    if not checkout.exists():
        return False
    try:
        head = _git(["-C", str(checkout), "rev-parse", "HEAD"], hooks=hooks)
        status = _git(
            [
                "-C",
                str(checkout),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignored",
            ],
            hooks=hooks,
        )
    except RepositoryAcquisitionError as error:
        raise RepositoryAcquisitionError(
            f"cached checkout is invalid: {checkout}"
        ) from error
    if head.lower() == commit and not status:
        return True
    if head.lower() != commit:
        raise RepositoryAcquisitionError(
            f"cached checkout HEAD does not match its commit path: {checkout}"
        )
    _git(["-C", str(checkout), "reset", "--hard", commit], hooks=hooks)
    _git(["-C", str(checkout), "clean", "-ffdx"], hooks=hooks)
    return True


def _create_checkout(
    mirror: Path,
    checkout: Path,
    commit: str,
    hooks: Path,
) -> None:
    checkout.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkout.with_name(f"{checkout.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        _git(
            ["clone", "--no-checkout", str(mirror), str(temporary)],
            hooks=hooks,
        )
        _git(
            ["-C", str(temporary), "checkout", "--detach", commit],
            hooks=hooks,
        )
        temporary.replace(checkout)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _is_bare_repository(repository: Path, hooks: Path) -> bool:
    try:
        return (
            _git(
                [f"--git-dir={repository}", "rev-parse", "--is-bare-repository"],
                hooks=hooks,
            )
            == "true"
        )
    except RepositoryAcquisitionError:
        return False


def _git(arguments: list[str], *, hooks: Path) -> str:
    command = [
        "git",
        "-c",
        f"core.hooksPath={hooks.resolve()}",
        "-c",
        "core.autocrlf=false",
        "-c",
        "protocol.ext.allow=never",
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise RepositoryAcquisitionError("git executable was not found") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "git command failed").strip()
        raise RepositoryAcquisitionError(detail) from error
    return completed.stdout.strip()
