"""Deterministic context construction from final expanded retrieval results."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.retrieval.contracts import canonical_chunk_id
from src.retrieval.relationship_expansion import ExpandedSearchResult, ResultOrigin


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One citation-addressable code chunk in a query-local context."""

    citation_id: str
    evidence_id: str
    source: str
    chunk_index: int
    page_content: str
    start_line: int | None
    end_line: int | None
    origin: ResultOrigin


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Ordered evidence and the exact text rendered from it."""

    evidence: tuple[EvidenceItem, ...]
    rendered_context: str


def build_context(
    results: Sequence[ExpandedSearchResult],
    *,
    max_chars: int | None = None,
) -> ContextBundle:
    """Deduplicate, cite, and render final results within an optional budget."""
    _validate_max_chars(max_chars)
    unique_results = _deduplicate(results)
    evidence: list[EvidenceItem] = []
    blocks: list[str] = []

    for index, result in enumerate(unique_results, start=1):
        item = _evidence_item(result, citation_id=f"C{index}")
        block = _render_evidence(item)
        rendered = "\n\n".join((*blocks, block))
        if max_chars is not None and len(rendered) > max_chars:
            break
        evidence.append(item)
        blocks.append(block)

    return ContextBundle(tuple(evidence), "\n\n".join(blocks))


def _deduplicate(
    results: Sequence[ExpandedSearchResult],
) -> list[ExpandedSearchResult]:
    unique: list[ExpandedSearchResult] = []
    seen: set[str] = set()
    for result in results:
        evidence_id = canonical_chunk_id(result.document)
        if evidence_id not in seen:
            unique.append(result)
            seen.add(evidence_id)
    return unique


def _evidence_item(
    result: ExpandedSearchResult,
    *,
    citation_id: str,
) -> EvidenceItem:
    document = result.document
    source = document.metadata["source"]
    chunk_index = document.metadata["chunk_index"]
    assert isinstance(source, str)
    assert isinstance(chunk_index, int)
    start_line = _optional_line(document.metadata.get("start_line"), "start_line")
    end_line = _optional_line(document.metadata.get("end_line"), "end_line")
    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError("start_line must not be greater than end_line")
    return EvidenceItem(
        citation_id=citation_id,
        evidence_id=canonical_chunk_id(document),
        source=source,
        chunk_index=chunk_index,
        page_content=document.page_content,
        start_line=start_line,
        end_line=end_line,
        origin=result.origin,
    )


def _render_evidence(item: EvidenceItem) -> str:
    location = item.source
    if item.start_line is not None and item.end_line is not None:
        location = f"{location}:{item.start_line}-{item.end_line}"
    return f"[{item.citation_id}] {location}\n{item.page_content}"


def _optional_line(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when present")
    return value


def _validate_max_chars(max_chars: int | None) -> None:
    if max_chars is None:
        return
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer or None")
