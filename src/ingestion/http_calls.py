"""Recognize conservative outbound HTTP calls from Tree-sitter syntax nodes."""

import re
from collections.abc import Iterator

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from src.ingestion.structure import HttpCall, SourceSpan

HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
_METHOD_SET = frozenset(HTTP_METHODS)
_IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"


def extract_http_calls(source: str, language: str) -> tuple[HttpCall, ...]:
    """Extract statically obvious outbound calls using an in-memory syntax parse."""
    if language not in {"python", "javascript", "typescript"}:
        return ()

    source_bytes = source.encode("utf-8")
    root = get_parser(language).parse(source_bytes).root_node
    if language == "python":
        calls = _extract_python_calls(root, source_bytes)
    else:
        calls = _extract_javascript_calls(root, source_bytes)
    return tuple(sorted(set(calls), key=_call_sort_key))


def _extract_javascript_calls(root: Node, source: bytes) -> list[HttpCall]:
    calls: list[HttpCall] = []
    for call in (node for node in _walk(root) if node.type == "call_expression"):
        function = call.child_by_field_name("function")
        arguments = call.child_by_field_name("arguments")
        if function is None or arguments is None or not arguments.named_children:
            continue
        target = _javascript_target(arguments.named_children[0], source)
        if target is None:
            continue

        client: str | None = None
        method: str | None = None
        if function.type == "identifier" and _text(function, source) == "fetch":
            client = "fetch"
            method = _fetch_method(arguments.named_children[1:], source)
        elif function.type == "member_expression":
            parts = function.named_children
            if len(parts) == 2 and _text(parts[0], source) == "axios":
                candidate = _text(parts[1], source).upper()
                if candidate in _METHOD_SET:
                    client = "axios"
                    method = candidate
        if client is None:
            continue
        calls.append(
            HttpCall(
                method=method,
                target=target,
                client=client,
                span=_span(call),
                caller=_javascript_caller(call, source),
            )
        )
    return calls


def _fetch_method(option_nodes: list[Node], source: bytes) -> str | None:
    if not option_nodes:
        return "GET"
    options = option_nodes[0]
    if options.type != "object":
        return None
    for pair in (child for child in options.named_children if child.type == "pair"):
        children = pair.named_children
        if len(children) != 2 or _text(children[0], source) != "method":
            continue
        value = _quoted_literal(children[1], source)
        if value is None:
            return None
        method = value.upper()
        return method if method in _METHOD_SET else None
    return "GET"


def _extract_python_calls(root: Node, source: bytes) -> list[HttpCall]:
    calls: list[HttpCall] = []
    for call in (node for node in _walk(root) if node.type == "call"):
        function = call.child_by_field_name("function")
        arguments = call.child_by_field_name("arguments")
        if (
            function is None
            or function.type != "attribute"
            or arguments is None
            or not arguments.named_children
        ):
            continue
        receiver = function.child_by_field_name("object")
        attribute = function.child_by_field_name("attribute")
        if receiver is None or attribute is None:
            continue
        client = _text(receiver, source)
        method = _text(attribute, source).upper()
        if client not in {"requests", "httpx"} or method not in _METHOD_SET:
            continue
        target = _python_target(arguments.named_children[0], source)
        if target is None:
            continue
        calls.append(
            HttpCall(
                method=method,
                target=target,
                client=client,
                span=_span(call),
                caller=_python_caller(call, source),
            )
        )
    return calls


def _javascript_target(node: Node, source: bytes) -> str | None:
    literal = _quoted_literal(node, source)
    if literal is not None:
        return literal
    if node.type != "template_string":
        return None
    raw = _text(node, source)
    if len(raw) < 2 or raw[0] != "`" or raw[-1] != "`":
        return None
    body = raw[1:-1]
    pattern = rf"\$\{{({_IDENTIFIER})\}}"
    remainder = re.sub(pattern, "", body)
    if "${" in remainder or "`" in remainder or "\\" in remainder:
        return None
    return re.sub(pattern, r"{\1}", body)


def _python_target(node: Node, source: bytes) -> str | None:
    literal = _quoted_literal(node, source)
    if literal is not None:
        return literal
    if node.type != "string":
        return None
    raw = _text(node, source)
    match = re.fullmatch(r"[fF](['\"])(.*)\1", raw, flags=re.DOTALL)
    if match is None:
        return None
    body = match.group(2)
    pattern = rf"\{{({_IDENTIFIER})\}}"
    remainder = re.sub(pattern, "", body)
    if "{" in remainder or "}" in remainder or "\\" in remainder:
        return None
    return re.sub(pattern, r"{\1}", body)


def _quoted_literal(node: Node, source: bytes) -> str | None:
    if node.type not in {"string", "string_literal"}:
        return None
    raw = _text(node, source)
    if len(raw) < 2 or raw[0] not in {'"', "'"} or raw[-1] != raw[0]:
        return None
    value = raw[1:-1]
    return None if "\\" in value else value


def _javascript_caller(node: Node, source: bytes) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type in {"function_declaration", "method_definition"}:
            return _field_text(parent, "name", source)
        parent = parent.parent
    return None


def _python_caller(node: Node, source: bytes) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type == "function_definition":
            return _field_text(parent, "name", source)
        parent = parent.parent
    return None


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _field_text(node: Node, field: str, source: bytes) -> str | None:
    child = node.child_by_field_name(field)
    return _text(child, source) if child is not None else None


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _span(node: Node) -> SourceSpan:
    return SourceSpan(
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
        start_column=node.start_point.column,
        end_column=node.end_point.column,
    )


def _call_sort_key(call: HttpCall) -> tuple[object, ...]:
    return (
        call.span.start_line,
        call.span.end_line,
        call.client,
        call.method or "",
        call.target,
        call.caller or "",
    )
