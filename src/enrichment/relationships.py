"""Link enriched outbound-call chunks to matching backend-route chunks."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

RELATED_ROUTE_CHUNKS_KEY = "related_route_chunks"
RELATED_HTTP_CALL_CHUNKS_KEY = "related_http_call_chunks"


class RelationshipLinkingError(ValueError):
    """Raised when chunk identity metadata cannot support stable relationships."""


@dataclass(frozen=True, order=True, slots=True)
class _ChunkReference:
    source: str
    chunk_index: int
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class _RouteOccurrence:
    reference: _ChunkReference
    path: str
    methods: tuple[str, ...]
    segments: tuple[str, ...]
    parameter_segments: tuple[bool, ...]

    @property
    def specificity(self) -> tuple[int, int]:
        parameter_count = sum(self.parameter_segments)
        return len(self.segments) - parameter_count, -parameter_count


@dataclass(frozen=True, slots=True)
class _CallOccurrence:
    reference: _ChunkReference
    method: str
    segments: tuple[str, ...]


def link_http_calls_to_routes(chunks: Sequence[Document]) -> list[Document]:
    """Return copies of enriched chunks with bidirectional HTTP relationships."""
    references = [_chunk_reference(chunk) for chunk in chunks]
    if len(set((item.source, item.chunk_index) for item in references)) != len(chunks):
        raise RelationshipLinkingError(
            "chunk source and chunk_index must uniquely identify the corpus"
        )

    routes = _route_occurrences(chunks, references)
    calls = _call_occurrences(chunks, references)
    route_links: dict[_ChunkReference, set[_ChunkReference]] = {
        reference: set() for reference in references
    }
    call_links: dict[_ChunkReference, set[_ChunkReference]] = {
        reference: set() for reference in references
    }

    for call in calls:
        candidates = [route for route in routes if _matches(call, route)]
        if not candidates:
            continue
        best_specificity = max(route.specificity for route in candidates)
        for route in candidates:
            if route.specificity != best_specificity:
                continue
            route_links[call.reference].add(route.reference)
            call_links[route.reference].add(call.reference)

    linked: list[Document] = []
    for chunk, reference in zip(chunks, references, strict=True):
        metadata = dict(chunk.metadata)
        metadata[RELATED_ROUTE_CHUNKS_KEY] = [
            _reference_metadata(item) for item in sorted(route_links[reference])
        ]
        metadata[RELATED_HTTP_CALL_CHUNKS_KEY] = [
            _reference_metadata(item) for item in sorted(call_links[reference])
        ]
        linked.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
                id=chunk.id,
            )
        )
    return linked


def _chunk_reference(chunk: Document) -> _ChunkReference:
    source = chunk.metadata.get("source")
    if not isinstance(source, str) or not source:
        raise RelationshipLinkingError("every chunk requires a non-empty string source")
    chunk_index = _required_reference_int(chunk.metadata, "chunk_index", 0, source)
    start_line = _required_reference_int(chunk.metadata, "start_line", 1, source)
    end_line = _required_reference_int(chunk.metadata, "end_line", 1, source)
    if start_line > end_line:
        raise RelationshipLinkingError(
            f"chunk {source!r} has invalid line range {start_line}-{end_line}"
        )
    return _ChunkReference(source, chunk_index, start_line, end_line)


def _required_reference_int(
    metadata: Mapping[str, Any], key: str, minimum: int, source: str
) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RelationshipLinkingError(
            f"chunk {source!r} requires valid integer {key!r} metadata"
        )
    return value


def _route_occurrences(
    chunks: Sequence[Document], references: Sequence[_ChunkReference]
) -> list[_RouteOccurrence]:
    occurrences: set[_RouteOccurrence] = set()
    for chunk, reference in zip(chunks, references, strict=True):
        for record in _metadata_records(chunk.metadata, "structural_routes"):
            path = record.get("path")
            methods_value = record.get("methods")
            if not isinstance(path, str) or not isinstance(methods_value, list):
                continue
            if not all(isinstance(method, str) for method in methods_value):
                continue
            pattern = _backend_pattern(path)
            if pattern is None:
                continue
            segments, parameters = pattern
            occurrences.add(
                _RouteOccurrence(
                    reference=reference,
                    path=path,
                    methods=tuple(methods_value),
                    segments=segments,
                    parameter_segments=parameters,
                )
            )
    return sorted(
        occurrences,
        key=lambda item: (
            item.reference,
            item.path,
            item.methods,
            item.segments,
        ),
    )


def _call_occurrences(
    chunks: Sequence[Document], references: Sequence[_ChunkReference]
) -> list[_CallOccurrence]:
    occurrences: set[_CallOccurrence] = set()
    for chunk, reference in zip(chunks, references, strict=True):
        for record in _metadata_records(chunk.metadata, "structural_http_calls"):
            method = record.get("method")
            target = record.get("target")
            if not isinstance(method, str) or not isinstance(target, str):
                continue
            segments = _local_path_segments(target)
            if segments is not None:
                occurrences.add(_CallOccurrence(reference, method, segments))
    return sorted(
        occurrences,
        key=lambda item: (item.reference, item.method, item.segments),
    )


def _metadata_records(
    metadata: Mapping[str, Any], key: str
) -> list[Mapping[str, Any]]:
    value = metadata.get(key, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _local_path_segments(target: str) -> tuple[str, ...] | None:
    if not target.startswith("/") or target.startswith("//"):
        return None
    path = target.split("#", 1)[0].split("?", 1)[0]
    return _path_segments(path)


def _backend_pattern(
    path: str,
) -> tuple[tuple[str, ...], tuple[bool, ...]] | None:
    if not path.startswith("/") or path.startswith("//"):
        return None
    normalized = path.split("#", 1)[0].split("?", 1)[0]
    segments = _path_segments(normalized)
    if segments is None:
        return None
    parameters: list[bool] = []
    for segment in segments:
        is_parameter = bool(
            re.fullmatch(r"\{[A-Za-z_][A-Za-z0-9_]*\}", segment)
            or re.fullmatch(r":[A-Za-z_][A-Za-z0-9_]*", segment)
        )
        if not is_parameter and (
            segment.startswith(":") or "{" in segment or "}" in segment
        ):
            return None
        parameters.append(is_parameter)
    return segments, tuple(parameters)


def _path_segments(path: str) -> tuple[str, ...] | None:
    if path == "/":
        return ()
    normalized = path[:-1] if path.endswith("/") else path
    segments = tuple(normalized[1:].split("/"))
    return None if any(not segment for segment in segments) else segments


def _matches(call: _CallOccurrence, route: _RouteOccurrence) -> bool:
    if route.methods and call.method not in route.methods:
        return False
    if len(call.segments) != len(route.segments):
        return False
    return all(
        is_parameter or call_segment == route_segment
        for call_segment, route_segment, is_parameter in zip(
            call.segments,
            route.segments,
            route.parameter_segments,
            strict=True,
        )
    )


def _reference_metadata(reference: _ChunkReference) -> dict[str, object]:
    return {
        "source": reference.source,
        "chunk_index": reference.chunk_index,
        "start_line": reference.start_line,
        "end_line": reference.end_line,
    }
