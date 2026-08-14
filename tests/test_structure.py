from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from src.ingestion.structure import StructuralExtractionError, extract_structure


def source_document(content: str, language: str, source: str = "sample") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "language": language},
    )


def test_extracts_python_definitions_imports_nested_method_and_spans() -> None:
    document = source_document(
        "import os\n"
        "from package import value\n"
        "\n"
        "def top():\n"
        "    return value\n"
        "\n"
        "class Service:\n"
        "    def method(self):\n"
        "        return os.getcwd()\n",
        "python",
        "src/service.py",
    )

    result = extract_structure(document)

    assert result.language == "python"
    assert result.source == "src/service.py"
    assert [item.source for item in result.imports] == [
        "import os",
        "from package import value",
    ]
    assert [definition.name for definition in result.definitions] == [
        "top",
        "Service",
        "method",
    ]
    method = result.definitions[2]
    assert method.parent == "Service"
    assert method.qualified_name == "Service.method"
    assert (method.span.start_line, method.span.end_line) == (8, 9)
    assert result.definitions[0].span.start_line == 4


def test_extracts_javascript_definitions_and_import() -> None:
    result = extract_structure(
        source_document(
            'import value from "package";\n'
            "function top() { return value; }\n"
            "class Service { method() { return value; } }\n",
            "javascript",
            "web/app.js",
        )
    )

    assert len(result.imports) == 1
    assert [item.qualified_name for item in result.definitions] == [
        "top",
        "Service",
        "Service.method",
    ]


def test_extracts_java_package_class_method_and_import() -> None:
    result = extract_structure(
        source_document(
            "package demo;\n"
            "import java.util.List;\n"
            "class Service {\n"
            "    int method() { return 1; }\n"
            "}\n",
            "java",
            "Service.java",
        )
    )

    assert [item.source for item in result.imports] == ["import java.util.List;"]
    assert [item.qualified_name for item in result.definitions] == [
        "demo",
        "Service",
        "Service.method",
    ]


def test_c_vendor_output_is_handled_without_inventing_names() -> None:
    result = extract_structure(
        source_document(
            "#include <stdio.h>\nint top(int value) { return value; }\n",
            "c",
            "main.c",
        )
    )

    # Version 1.14.x identifies the function kind/span but supplies no name and
    # does not classify preprocessor includes as imports. The adapter does not
    # fabricate semantic data that the structural API did not return.
    assert result.definitions == ()
    assert result.imports == ()


def test_nested_output_is_deterministic() -> None:
    document = source_document(
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            pass\n",
        "python",
    )

    first = extract_structure(document)
    second = extract_structure(document)

    assert first == second
    assert [item.qualified_name for item in first.definitions] == [
        "Outer",
        "Outer.Inner",
        "Outer.Inner.method",
    ]


def test_empty_source_has_no_structure() -> None:
    result = extract_structure(source_document("", "python"))

    assert result.definitions == ()
    assert result.imports == ()


def test_unsupported_language_fails_with_context() -> None:
    with pytest.raises(
        StructuralExtractionError,
        match=r"unknown\.ext.*language='definitely-not-a-language'.*unsupported",
    ):
        extract_structure(
            source_document("content", "definitely-not-a-language", "unknown.ext")
        )


def test_vendor_failure_is_wrapped_with_source_and_language() -> None:
    with patch("src.ingestion.structure.process", side_effect=RuntimeError("broken")):
        with pytest.raises(
            StructuralExtractionError,
            match=r"src/main\.py.*language='python'.*broken",
        ):
            extract_structure(source_document("pass\n", "python", "src/main.py"))
