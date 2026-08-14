"""Build deterministic structural text for the shared retrieval corpus."""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document

RAW_CONTENT_METADATA_KEY = "raw_content"


def enrich_retrieval_content(document: Document) -> Document:
    """Return a copy whose page content is structural summary plus exact raw code."""
    existing_raw = document.metadata.get(RAW_CONTENT_METADATA_KEY)
    raw_content = (
        existing_raw if isinstance(existing_raw, str) else document.page_content
    )
    metadata = dict(document.metadata)
    metadata[RAW_CONTENT_METADATA_KEY] = raw_content
    return Document(
        page_content=render_retrieval_content(raw_content, metadata),
        metadata=metadata,
        id=document.id,
    )


def enrich_retrieval_corpus(documents: Sequence[Document]) -> list[Document]:
    """Build retrieval representations without changing corpus order."""
    return [enrich_retrieval_content(document) for document in documents]


def render_retrieval_content(raw_content: str, metadata: Mapping[str, Any]) -> str:
    """Render supported structural fields in a compact, stable section order."""
    sections: list[str] = []
    source = metadata.get("source")
    if isinstance(source, str) and source:
        sections.append(f"Source: {source}")

    _append_section(
        sections,
        "Definitions",
        _render_records(metadata.get("structural_definitions"), _definition_line),
    )
    _append_section(
        sections,
        "Imports",
        _render_records(metadata.get("structural_imports"), _import_line),
    )
    _append_section(
        sections,
        "Routes",
        _render_records(metadata.get("structural_routes"), _route_line),
    )
    _append_section(
        sections,
        "Outbound HTTP Calls",
        _render_records(metadata.get("structural_http_calls"), _http_call_line),
    )
    sections.append(f"Code:\n{raw_content}")
    return "\n\n".join(sections)


def _append_section(sections: list[str], label: str, lines: list[str]) -> None:
    if lines:
        sections.append(f"{label}:\n" + "\n".join(f"- {line}" for line in lines))


def _render_records(value: object, renderer: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    rendered: list[tuple[int, int, str]] = []
    for record in value:
        if not isinstance(record, Mapping):
            continue
        line = renderer(record)
        if line is None:
            continue
        rendered.append(
            (
                _sort_integer(record.get("start_line")),
                _sort_integer(record.get("end_line")),
                line,
            )
        )
    return [line for _, _, line in sorted(rendered)]


def _definition_line(record: Mapping[str, object]) -> str | None:
    name = _text(record.get("qualified_name")) or _text(record.get("name"))
    if name is None:
        return None
    details: list[str] = []
    _add_detail(details, "Kind", _text(record.get("kind")))
    _add_detail(details, "Signature", _text(record.get("signature")))
    return _with_details(name, details)


def _import_line(record: Mapping[str, object]) -> str | None:
    source = _text(record.get("source"))
    if source is None:
        return None
    details: list[str] = []
    items = record.get("items")
    if isinstance(items, list):
        item_text = ", ".join(item for item in items if isinstance(item, str) and item)
        _add_detail(details, "Items", item_text or None)
    _add_detail(details, "Alias", _text(record.get("alias")))
    if record.get("is_wildcard") is True:
        details.append("Wildcard: yes")
    return _with_details(source, details)


def _route_line(record: Mapping[str, object]) -> str | None:
    path = _text(record.get("path"))
    if path is None:
        return None
    methods = record.get("methods")
    method_text = ""
    if isinstance(methods, list):
        method_text = ", ".join(
            method for method in methods if isinstance(method, str) and method
        )
    summary = f"{method_text} {path}" if method_text else path
    details: list[str] = []
    _add_detail(details, "Framework", _text(record.get("framework")))
    _add_detail(details, "Handler", _text(record.get("handler")))
    _add_detail(details, "Owner", _text(record.get("owner")))
    return _with_details(summary, details)


def _http_call_line(record: Mapping[str, object]) -> str | None:
    target = _text(record.get("target"))
    if target is None:
        return None
    method = _text(record.get("method"))
    summary = f"{method} {target}" if method else target
    details: list[str] = []
    _add_detail(details, "Client", _text(record.get("client")))
    _add_detail(details, "Caller", _text(record.get("caller")))
    return _with_details(summary, details)


def _with_details(summary: str, details: Sequence[str]) -> str:
    return summary if not details else f"{summary} | " + " | ".join(details)


def _add_detail(details: list[str], label: str, value: str | None) -> None:
    if value is not None:
        details.append(f"{label}: {value}")


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sort_integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 2**63
