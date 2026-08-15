"""Discover candidate source files within a repository."""

import os
from pathlib import Path

SUPPORTED_SOURCE_EXTENSIONS = frozenset(
    {
        ".bash",
        ".c",
        ".cc",
        ".cjs",
        ".cpp",
        ".cs",
        ".cxx",
        ".go",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".mjs",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".sh",
        ".sql",
        ".swift",
        ".ts",
        ".tsx",
        ".zsh",
    }
)

IGNORED_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".gradle",
        ".hg",
        ".idea",
        ".mypy_cache",
        ".next",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "__pypackages__",
        "build",
        "coverage",
        "dist",
        "env",
        "node_modules",
        "site-packages",
        "target",
        "vendor",
        "venv",
    }
)


def find_source_files(repo_path: Path) -> list[Path]:
    """Return supported source files as sorted paths relative to ``repo_path``."""
    if not repo_path.exists():
        raise FileNotFoundError(repo_path)
    if not repo_path.is_dir():
        raise NotADirectoryError(repo_path)

    source_files: list[Path] = []

    for current_root, directory_names, file_names in os.walk(
        repo_path, followlinks=False
    ):
        current_path = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name.casefold() not in IGNORED_DIRECTORIES
            and not (current_path / name).is_symlink()
        )

        for file_name in file_names:
            file_path = current_path / file_name
            if (
                file_path.suffix.casefold() in SUPPORTED_SOURCE_EXTENSIONS
                and not file_path.is_symlink()
            ):
                source_files.append(file_path.relative_to(repo_path))

    return sorted(source_files, key=lambda path: path.as_posix())

