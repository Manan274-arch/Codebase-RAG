"""Map whole-file structural records onto code chunks by source-line overlap."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from langchain_core.documents import Document

from src.enrichment.structure import (
    Definition,
    FileStructure,
    HttpCall,
    Import,
    RouteDefinition,
)

DEFINITIONS_METADATA_KEY = "structural_definitions"
IMPORTS_METADATA_KEY = "structural_imports"
ROUTES_METADATA_KEY = "structural_routes"
HTTP_CALLS_METADATA_KEY = "structural_http_calls"


class StructuralEnrichmentError(ValueError):
    """Raised when chunks cannot be safely matched to a file structure."""


def line_ranges_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    """Return whether two one-based, inclusive line ranges overlap."""
    return max(first_start, second_start) <= min(first_end, second_end)


def enrich_chunks(
    chunks: Sequence[Document], file_structure: FileStructure
) -> list[Document]:
    """Return metadata-enriched copies of chunks for one source file."""
    definitions = _ordered_unique_definitions(file_structure.definitions)
    imports = _ordered_unique_imports(file_structure.imports)
    routes = _ordered_unique_routes(file_structure.routes)
    http_calls = _ordered_unique_http_calls(file_structure.http_calls)
    enriched: list[Document] = []

    for chunk in chunks:
        source = chunk.metadata.get("source")
        if file_structure.source is not None and source != file_structure.source:
            raise StructuralEnrichmentError(
                "cannot enrich chunk from "
                f"{source!r} with structure from {file_structure.source!r}"
            )

        start_line = _required_line(chunk.metadata, "start_line", source)
        end_line = _required_line(chunk.metadata, "end_line", source)
        if start_line > end_line:
            raise StructuralEnrichmentError(
                f"invalid chunk line range {start_line}-{end_line} "
                f"for {source or '<unknown source>'}"
            )

        metadata = dict(chunk.metadata)
        metadata[DEFINITIONS_METADATA_KEY] = [
            _definition_metadata(definition)
            for definition in definitions
            if line_ranges_overlap(
                start_line,
                end_line,
                definition.span.start_line,
                definition.span.end_line,
            )
        ]
        metadata[IMPORTS_METADATA_KEY] = [
            _import_metadata(item)
            for item in imports
            if line_ranges_overlap(
                start_line,
                end_line,
                item.span.start_line,
                item.span.end_line,
            )
        ]
        metadata[ROUTES_METADATA_KEY] = [
            _route_metadata(route)
            for route in routes
            if line_ranges_overlap(
                start_line,
                end_line,
                route.span.start_line,
                route.span.end_line,
            )
        ]
        metadata[HTTP_CALLS_METADATA_KEY] = [
            _http_call_metadata(call)
            for call in http_calls
            if line_ranges_overlap(
                start_line,
                end_line,
                call.span.start_line,
                call.span.end_line,
            )
        ]
        enriched.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
                id=chunk.id,
            )
        )

    return enriched


def _required_line(
    metadata: Mapping[str, Any], key: str, source: object
) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise StructuralEnrichmentError(
            f"chunk for {source or '<unknown source>'} requires a positive "
            f"integer {key!r}"
        )
    return value


def _ordered_unique_definitions(
    definitions: Iterable[Definition],
) -> list[Definition]:
    ordered = sorted(
        definitions,
        key=lambda item: (
            item.span.start_line,
            item.span.end_line,
            item.kind,
            item.qualified_name or item.name,
            item.signature or "",
        ),
    )
    return list(dict.fromkeys(ordered))


def _ordered_unique_imports(imports: Iterable[Import]) -> list[Import]:
    ordered = sorted(
        imports,
        key=lambda item: (
            item.span.start_line,
            item.span.end_line,
            item.source,
            item.items,
            item.alias or "",
            item.is_wildcard,
        ),
    )
    return list(dict.fromkeys(ordered))


def _ordered_unique_routes(routes: Iterable[RouteDefinition]) -> list[RouteDefinition]:
    ordered = sorted(
        routes,
        key=lambda item: (
            item.span.start_line,
            item.span.end_line,
            item.path,
            item.methods,
            item.framework,
            item.handler or "",
            item.owner or "",
        ),
    )
    return list(dict.fromkeys(ordered))


def _ordered_unique_http_calls(calls: Iterable[HttpCall]) -> list[HttpCall]:
    ordered = sorted(
        calls,
        key=lambda item: (
            item.span.start_line,
            item.span.end_line,
            item.client,
            item.method or "",
            item.target,
            item.caller or "",
        ),
    )
    return list(dict.fromkeys(ordered))


def _definition_metadata(definition: Definition) -> dict[str, object]:
    return {
        "name": definition.name,
        "qualified_name": definition.qualified_name,
        "kind": definition.kind,
        "start_line": definition.span.start_line,
        "end_line": definition.span.end_line,
        "signature": definition.signature,
    }


def _import_metadata(item: Import) -> dict[str, object]:
    return {
        "source": item.source,
        "items": list(item.items),
        "alias": item.alias,
        "is_wildcard": item.is_wildcard,
        "start_line": item.span.start_line,
        "end_line": item.span.end_line,
    }


def _route_metadata(route: RouteDefinition) -> dict[str, object]:
    return {
        "path": route.path,
        "methods": list(route.methods),
        "framework": route.framework,
        "handler": route.handler,
        "owner": route.owner,
        "start_line": route.span.start_line,
        "end_line": route.span.end_line,
    }


def _http_call_metadata(call: HttpCall) -> dict[str, object]:
    return {
        "method": call.method,
        "target": call.target,
        "client": call.client,
        "caller": call.caller,
        "start_line": call.span.start_line,
        "end_line": call.span.end_line,
    }
