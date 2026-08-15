"""Recognize a small matrix of backend routes from Tree-sitter syntax nodes."""

from collections.abc import Iterator

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from src.enrichment.structure import RouteDefinition, SourceSpan

HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD")
_METHOD_SET = frozenset(HTTP_METHODS)
_SPRING_METHOD_ANNOTATIONS = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "PatchMapping": "PATCH",
    "DeleteMapping": "DELETE",
}


def extract_routes(source: str, language: str) -> tuple[RouteDefinition, ...]:
    """Extract statically obvious server routes using one in-memory syntax parse."""
    if language not in {"python", "javascript", "typescript", "java"}:
        return ()

    source_bytes = source.encode("utf-8")
    root = get_parser(language).parse(source_bytes).root_node
    if language == "python":
        routes = _extract_python_routes(root, source_bytes)
    elif language in {"javascript", "typescript"}:
        routes = _extract_express_routes(root, source_bytes)
    else:
        routes = _extract_spring_routes(root, source_bytes)

    return tuple(sorted(routes, key=_route_sort_key))


def _extract_python_routes(root: Node, source: bytes) -> list[RouteDefinition]:
    flask_imported = any(
        "flask" in _text(node, source).casefold()
        for node in _walk(root)
        if node.type in {"import_statement", "import_from_statement"}
    )
    routes: list[RouteDefinition] = []
    decorated_nodes = (
        node for node in _walk(root) if node.type == "decorated_definition"
    )
    for decorated in decorated_nodes:
        function = next(
            (
                child
                for child in decorated.named_children
                if child.type == "function_definition"
            ),
            None,
        )
        if function is None:
            continue
        handler = _field_text(function, "name", source)
        for decorator in (
            child for child in decorated.named_children if child.type == "decorator"
        ):
            route = _python_decorator_route(
                decorator, decorated, handler, flask_imported, source
            )
            if route is not None:
                routes.append(route)
    return routes


def _python_decorator_route(
    decorator: Node,
    decorated: Node,
    handler: str | None,
    flask_imported: bool,
    source: bytes,
) -> RouteDefinition | None:
    call = next(
        (child for child in decorator.named_children if child.type == "call"),
        None,
    )
    if call is None:
        return None
    attribute = call.child_by_field_name("function")
    arguments = call.child_by_field_name("arguments")
    if attribute is None or attribute.type != "attribute" or arguments is None:
        return None

    receiver = _field_text(attribute, "object", source)
    method_node = attribute.child_by_field_name("attribute")
    method = _text(method_node, source).casefold() if method_node is not None else ""
    if receiver not in {"app", "router", "blueprint"}:
        return None

    argument_nodes = arguments.named_children
    path = _first_static_string(argument_nodes, source)
    if path is None:
        return None

    if method == "route":
        methods = _python_methods_keyword(argument_nodes, source) or ("GET",)
        framework = "flask"
    elif method.upper() in _METHOD_SET:
        methods = (method.upper(),)
        framework = "flask" if flask_imported else "fastapi"
    else:
        return None

    return RouteDefinition(
        path=path,
        methods=methods,
        framework=framework,
        handler=handler,
        span=_span(decorated),
        owner=receiver,
    )


def _python_methods_keyword(nodes: list[Node], source: bytes) -> tuple[str, ...]:
    for node in nodes:
        if node.type != "keyword_argument" or not node.named_children:
            continue
        if _text(node.named_children[0], source) != "methods":
            continue
        methods = {
            value.upper()
            for child in _walk(node)
            if (value := _static_string(child, source)) is not None
            and value.upper() in _METHOD_SET
        }
        return _ordered_methods(methods)
    return ()


def _extract_express_routes(root: Node, source: bytes) -> list[RouteDefinition]:
    routes: list[RouteDefinition] = []
    for call in (node for node in _walk(root) if node.type == "call_expression"):
        function = call.child_by_field_name("function")
        arguments = call.child_by_field_name("arguments")
        if (
            function is None
            or function.type != "member_expression"
            or arguments is None
        ):
            continue
        parts = function.named_children
        if len(parts) != 2:
            continue
        receiver = _text(parts[0], source)
        method = _text(parts[1], source).upper()
        if receiver not in {"app", "router"} or method not in _METHOD_SET:
            continue
        argument_nodes = arguments.named_children
        path = _first_static_string(argument_nodes, source)
        if path is None:
            continue
        handler = None
        if len(argument_nodes) > 1 and argument_nodes[1].type == "identifier":
            handler = _text(argument_nodes[1], source)
        statement = (
            call.parent
            if call.parent and call.parent.type == "expression_statement"
            else call
        )
        routes.append(
            RouteDefinition(
                path=path,
                methods=(method,),
                framework="express",
                handler=handler,
                span=_span(statement),
                owner=receiver,
            )
        )
    return routes


def _extract_spring_routes(root: Node, source: bytes) -> list[RouteDefinition]:
    routes: list[RouteDefinition] = []
    for method in (node for node in _walk(root) if node.type == "method_declaration"):
        owner_node = _nearest_parent(method, "class_declaration")
        owner = _field_text(owner_node, "name", source) if owner_node else None
        prefix = _spring_class_prefix(owner_node, source) if owner_node else ""
        handler = _field_text(method, "name", source)
        for annotation in _direct_annotations(method):
            mapping = _spring_mapping(annotation, source)
            if mapping is None:
                continue
            path, methods = mapping
            routes.append(
                RouteDefinition(
                    path=_join_paths(prefix, path),
                    methods=methods,
                    framework="spring",
                    handler=handler,
                    span=_span(method),
                    owner=owner,
                )
            )
    return routes


def _spring_class_prefix(class_node: Node, source: bytes) -> str:
    for annotation in _direct_annotations(class_node):
        name = _field_text(annotation, "name", source)
        if name == "RequestMapping":
            mapping = _spring_mapping(annotation, source)
            if mapping is not None:
                return mapping[0]
    return ""


def _spring_mapping(
    annotation: Node, source: bytes
) -> tuple[str, tuple[str, ...]] | None:
    name = _field_text(annotation, "name", source)
    if name not in {*_SPRING_METHOD_ANNOTATIONS, "RequestMapping"}:
        return None
    path = _annotation_path(annotation, source)
    if path is None:
        return None
    if name in _SPRING_METHOD_ANNOTATIONS:
        return path, (_SPRING_METHOD_ANNOTATIONS[name],)

    methods: set[str] = set()
    pairs = (
        node for node in _walk(annotation) if node.type == "element_value_pair"
    )
    for pair in pairs:
        if not pair.named_children or _text(pair.named_children[0], source) != "method":
            continue
        for access in (node for node in _walk(pair) if node.type == "field_access"):
            value = _text(access.named_children[-1], source).upper()
            if value in _METHOD_SET:
                methods.add(value)
    return path, _ordered_methods(methods)


def _annotation_path(annotation: Node, source: bytes) -> str | None:
    pairs = [node for node in _walk(annotation) if node.type == "element_value_pair"]
    for pair in pairs:
        if not pair.named_children:
            continue
        if _text(pair.named_children[0], source) in {"path", "value"}:
            return _first_static_string(pair.named_children[1:], source)
    arguments = next(
        (
            child
            for child in annotation.named_children
            if child.type == "annotation_argument_list"
        ),
        None,
    )
    return _first_static_string(arguments.named_children, source) if arguments else None


def _direct_annotations(declaration: Node) -> list[Node]:
    modifiers = next(
        (child for child in declaration.named_children if child.type == "modifiers"),
        None,
    )
    if modifiers is None:
        return []
    return [
        child
        for child in modifiers.named_children
        if child.type in {"annotation", "marker_annotation"}
    ]


def _first_static_string(nodes: list[Node], source: bytes) -> str | None:
    for node in nodes:
        value = _static_string(node, source)
        if value is not None:
            return value
    return None


def _static_string(node: Node, source: bytes) -> str | None:
    if node.type not in {"string", "string_literal"}:
        return None
    raw = _text(node, source)
    if len(raw) < 2 or raw[0] not in {'"', "'"} or raw[-1] != raw[0]:
        return None
    value = raw[1:-1]
    return None if "\\" in value else value


def _ordered_methods(methods: set[str]) -> tuple[str, ...]:
    return tuple(method for method in HTTP_METHODS if method in methods)


def _join_paths(prefix: str, path: str) -> str:
    if not prefix:
        return path
    if not path:
        return prefix
    return f"/{prefix.strip('/')}/{path.strip('/')}"


def _nearest_parent(node: Node, node_type: str) -> Node | None:
    parent = node.parent
    while parent is not None:
        if parent.type == node_type:
            return parent
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


def _route_sort_key(route: RouteDefinition) -> tuple[object, ...]:
    return (
        route.span.start_line,
        route.span.end_line,
        route.path,
        route.methods,
        route.framework,
        route.handler or "",
    )
