"""Extract lightweight source structure from whole-file documents.

Public spans use one-based, inclusive line numbers. Columns remain zero-based,
matching Tree-sitter, and the end column is exclusive.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from langchain_core.documents import Document
from tree_sitter_language_pack import (
    ProcessConfig,
    has_language,
    process,
)


class _SpanLike(Protocol):
    start_line: int
    end_line: int
    start_column: int
    end_column: int


class _StructureItemLike(Protocol):
    kind: object
    name: str | None
    span: _SpanLike | None
    signature: str | None
    children: Sequence["_StructureItemLike"]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A source range with one-based inclusive lines and zero-based columns."""

    start_line: int
    end_line: int
    start_column: int
    end_column: int


@dataclass(frozen=True, slots=True)
class Definition:
    """A named definition discovered through lexical syntax structure."""

    name: str
    kind: str
    span: SourceSpan
    signature: str | None = None
    parent: str | None = None
    qualified_name: str | None = None


@dataclass(frozen=True, slots=True)
class Import:
    """A lightweight, unresolved import/include declaration."""

    source: str
    items: tuple[str, ...]
    alias: str | None
    is_wildcard: bool
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class RouteDefinition:
    """A statically recognizable server-side route declaration."""

    path: str
    methods: tuple[str, ...]
    framework: str
    handler: str | None
    span: SourceSpan
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class FileStructure:
    """Lightweight syntax records extracted from one complete source file."""

    language: str
    definitions: tuple[Definition, ...]
    imports: tuple[Import, ...]
    source: str | None = None
    routes: tuple[RouteDefinition, ...] = ()


class StructuralExtractionError(RuntimeError):
    """Raised when a whole-file structural analysis cannot be completed."""


def extract_structure(document: Document) -> FileStructure:
    """Extract definitions and imports from a whole-file LangChain document."""
    language_value = document.metadata.get("language")
    source_value = document.metadata.get("source")
    source = source_value if isinstance(source_value, str) else None

    if not isinstance(language_value, str) or not language_value:
        raise _error(source, language_value, "missing or invalid language metadata")

    language = _tree_sitter_language(language_value)
    try:
        if not has_language(language):
            raise _error(source, language, "unsupported language")
        result = process(
            document.page_content,
            ProcessConfig(
                language=language,
                structure=True,
                imports=True,
                exports=False,
                comments=False,
                docstrings=False,
                symbols=False,
                diagnostics=False,
                chunk_max_size=None,
                data_extraction=False,
            ),
        )
        vendor_structure = cast(Sequence[_StructureItemLike], result.structure)
        definitions = tuple(_flatten_definitions(vendor_structure))
        imports = tuple(
            Import(
                source=item.source,
                items=tuple(item.items),
                alias=item.alias,
                is_wildcard=item.is_wildcard,
                span=_convert_span(item.span),
            )
            for item in result.imports
            if item.span is not None
        )
        from src.ingestion.routes import extract_routes

        routes = extract_routes(document.page_content, language)
    except StructuralExtractionError:
        raise
    except Exception as error:
        raise _error(source, language, str(error)) from error

    return FileStructure(
        language=language,
        definitions=definitions,
        imports=imports,
        source=source,
        routes=routes,
    )


def _tree_sitter_language(language: str) -> str:
    """Adapt the project's normalized language label to the vendor identifier."""
    return language.casefold()


def _flatten_definitions(
    items: Sequence[_StructureItemLike], ancestry: tuple[str, ...] = ()
) -> list[Definition]:
    definitions: list[Definition] = []
    for item in items:
        name = item.name
        next_ancestry = ancestry
        if isinstance(name, str) and name and item.span is not None:
            parent = ".".join(ancestry) or None
            qualified_name = ".".join((*ancestry, name))
            definitions.append(
                Definition(
                    name=name,
                    kind=str(item.kind).casefold(),
                    span=_convert_span(item.span),
                    signature=item.signature,
                    parent=parent,
                    qualified_name=qualified_name,
                )
            )
            next_ancestry = (*ancestry, name)
        definitions.extend(_flatten_definitions(item.children, next_ancestry))
    return definitions


def _convert_span(span: _SpanLike) -> SourceSpan:
    return SourceSpan(
        start_line=span.start_line + 1,
        end_line=span.end_line + 1,
        start_column=span.start_column,
        end_column=span.end_column,
    )


def _error(
    source: str | None, language: object, reason: str
) -> StructuralExtractionError:
    return StructuralExtractionError(
        f"structural extraction failed for {source or '<unknown source>'} "
        f"(language={language!r}): {reason}"
    )
