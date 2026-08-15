import pytest
from langchain_core.documents import Document
from src.generation.context import build_context
from src.retrieval.relationship_expansion import (
    ExpandedSearchResult,
    ResultOrigin,
)


def result(
    source: str,
    chunk_index: int,
    content: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    origin: ResultOrigin = "retrieved",
) -> ExpandedSearchResult:
    metadata: dict[str, object] = {
        "source": source,
        "chunk_index": chunk_index,
    }
    if start_line is not None:
        metadata["start_line"] = start_line
    if end_line is not None:
        metadata["end_line"] = end_line
    document = Document(page_content=content, metadata=metadata)
    return ExpandedSearchResult(
        document=document,
        origin=origin,
        original_rank=1 if origin == "retrieved" else None,
        rerank_score=0.9 if origin == "retrieved" else None,
        first_stage_score=None,
        first_stage_rank=1 if origin == "retrieved" else None,
    )


def test_preserves_canonical_identity_and_assigns_local_citations() -> None:
    bundle = build_context(
        [
            result(
                "src/api/users.py",
                4,
                "def users(): pass",
                start_line=42,
                end_line=67,
            ),
            result("src/services/auth.py", 2, "def auth(): pass"),
        ]
    )

    assert [item.citation_id for item in bundle.evidence] == ["C1", "C2"]
    assert [item.evidence_id for item in bundle.evidence] == [
        "src/api/users.py::4",
        "src/services/auth.py::2",
    ]
    assert bundle.evidence[0].source == "src/api/users.py"
    assert bundle.evidence[0].chunk_index == 4


def test_deduplicates_by_canonical_identity_and_keeps_first_occurrence() -> None:
    duplicate = result("same.py", 0, "later duplicate", origin="relationship")
    bundle = build_context(
        [
            result("first.py", 0, "first"),
            result("same.py", 0, "original"),
            duplicate,
            result("last.py", 1, "last"),
        ]
    )

    assert [item.evidence_id for item in bundle.evidence] == [
        "first.py::0",
        "same.py::0",
        "last.py::1",
    ]
    assert [item.citation_id for item in bundle.evidence] == ["C1", "C2", "C3"]
    assert bundle.evidence[1].page_content == "original"
    assert bundle.evidence[1].origin == "retrieved"


def test_renders_source_ranges_and_missing_ranges_without_inventing_lines() -> None:
    bundle = build_context(
        [
            result("src/api/users.py", 0, "users code", start_line=42, end_line=67),
            result("src/services/auth.py", 0, "auth code"),
        ]
    )

    assert bundle.rendered_context == (
        "[C1] src/api/users.py:42-67\n"
        "users code\n\n"
        "[C2] src/services/auth.py\n"
        "auth code"
    )
    assert bundle.evidence[1].start_line is None
    assert bundle.evidence[1].end_line is None


def test_page_content_is_preserved_exactly_and_rendering_is_deterministic() -> None:
    content = "\n  def exact():\n      return True\n"
    results = [result("exact.py", 0, content, start_line=2, end_line=3)]

    first = build_context(results)
    second = build_context(results)

    assert first == second
    assert first.evidence[0].page_content == content
    assert first.rendered_context == f"[C1] exact.py:2-3\n{content}"


def test_no_budget_includes_all_deduplicated_evidence() -> None:
    results = [result(f"{name}.py", 0, name) for name in ("a", "b", "c")]

    bundle = build_context(results, max_chars=None)

    assert len(bundle.evidence) == 3


def test_character_budget_keeps_only_complete_rank_preserving_blocks() -> None:
    results = [result("a.py", 0, "alpha"), result("b.py", 0, "beta")]
    first_only = build_context(results[:1])

    exact = build_context(results, max_chars=len(first_only.rendered_context))
    too_small = build_context(results, max_chars=len(first_only.rendered_context) - 1)

    assert exact == first_only
    assert [item.citation_id for item in exact.evidence] == ["C1"]
    assert too_small.evidence == ()
    assert too_small.rendered_context == ""


@pytest.mark.parametrize("max_chars", [-1, 1.5, True])
def test_invalid_character_budget_fails_clearly(max_chars: object) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        build_context([], max_chars=max_chars)  # type: ignore[arg-type]


def test_invalid_optional_line_metadata_fails_clearly() -> None:
    malformed = result("bad.py", 0, "code", start_line=3, end_line=2)

    with pytest.raises(ValueError, match="start_line"):
        build_context([malformed])
