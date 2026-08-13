from pathlib import Path

import pytest
from src.ingestion.repository import find_source_files


def create_files(root: Path, relative_paths: list[str]) -> None:
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_finds_representative_source_files_recursively(tmp_path: Path) -> None:
    source_paths = [
        "app/main.py",
        "backend/Main.java",
        "native/program.c",
        "native/include/program.h",
        "native/engine.cpp",
        "native/include/engine.hpp",
        "web/index.js",
        "web/component.tsx",
        "cmd/server.go",
        "crates/core/lib.rs",
    ]
    create_files(tmp_path, source_paths)

    assert find_source_files(tmp_path) == [
        Path(path) for path in sorted(source_paths)
    ]


def test_excludes_unsupported_files(tmp_path: Path) -> None:
    create_files(
        tmp_path,
        ["README.md", "image.png", "data.csv", "bin/program", "src/program.py"],
    )

    assert find_source_files(tmp_path) == [Path("src/program.py")]


@pytest.mark.parametrize(
    "ignored_directory",
    [".git", "node_modules", "__pycache__", "build", "dist", ".venv"],
)
def test_prunes_ignored_directories(
    tmp_path: Path, ignored_directory: str
) -> None:
    create_files(
        tmp_path,
        [f"{ignored_directory}/nested/ignored.py", "src/included.py"],
    )

    assert find_source_files(tmp_path) == [Path("src/included.py")]


def test_returns_relative_paths_in_deterministic_order(tmp_path: Path) -> None:
    create_files(tmp_path, ["z.py", "nested/b.ts", "nested/a.java", "a.rs"])

    result = find_source_files(tmp_path)

    assert result == [
        Path("a.rs"),
        Path("nested/a.java"),
        Path("nested/b.ts"),
        Path("z.py"),
    ]
    assert all(not path.is_absolute() for path in result)


def test_raises_for_nonexistent_repository(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_source_files(tmp_path / "missing")


def test_raises_when_repository_path_is_a_file(tmp_path: Path) -> None:
    file_path = tmp_path / "file.py"
    file_path.touch()

    with pytest.raises(NotADirectoryError):
        find_source_files(file_path)


def test_does_not_recurse_through_symlinked_directories(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    external = tmp_path / "external"
    repository.mkdir()
    create_files(external, ["linked.py"])

    try:
        (repository / "linked").symlink_to(external, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Directory symlinks are unavailable: {error}")

    assert find_source_files(repository) == []


def test_matches_extensions_case_insensitively(tmp_path: Path) -> None:
    create_files(tmp_path, ["module.PY", "component.TsX", "README.MD"])

    assert find_source_files(tmp_path) == [Path("component.TsX"), Path("module.PY")]
