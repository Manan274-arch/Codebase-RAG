from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from groq import APIConnectionError, APIError, AuthenticationError, RateLimitError
from src.generation.context import ContextBundle, EvidenceItem
from src.generation.generator import CitationValidationError, generate_answer
from src.generation.groq import (
    DEFAULT_GROQ_MAX_TOKENS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GROQ_TEMPERATURE,
    GenerationBackendError,
    GroqTextGenerator,
)


@pytest.fixture(autouse=True)
def isolate_repository_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import config

    monkeypatch.setattr(config, "_REPOSITORY_ENV_FILE", tmp_path / ".env")


class FakeCompletions:
    def __init__(self, content: str | None = "answer") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.empty_choices = False
        self.malformed_choice = False

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.empty_choices:
            return SimpleNamespace(choices=[])
        if self.malformed_choice:
            return SimpleNamespace(choices=[SimpleNamespace()])
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeGroqFactory:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeClient:
        self.calls.append(kwargs)
        return FakeClient(self.completions)


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None = "answer",
) -> tuple[FakeGroqFactory, FakeCompletions]:
    from src.generation import groq as groq_module

    completions = FakeCompletions(content)
    factory = FakeGroqFactory(completions)
    monkeypatch.setattr(groq_module, "Groq", factory)
    return factory, completions


def context_bundle() -> ContextBundle:
    item = EvidenceItem(
        citation_id="C1",
        evidence_id="src/example.py::0",
        source="src/example.py",
        chunk_index=0,
        page_content="def answer(): return 42",
        start_line=1,
        end_line=1,
        origin="retrieved",
    )
    rendered = "[C1] src/example.py:1-1\ndef answer(): return 42"
    return ContextBundle((item,), rendered)


def test_default_model_and_generation_parameters_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, completions = install_fake_client(monkeypatch, "grounded [C1]")

    output = GroqTextGenerator(api_key="test-key").generate("exact prompt")

    assert factory.calls == [{"api_key": "test-key", "max_retries": 0}]
    assert completions.calls == [
        {
            "messages": [{"role": "user", "content": "exact prompt"}],
            "model": DEFAULT_GROQ_MODEL,
            "temperature": DEFAULT_GROQ_TEMPERATURE,
            "max_tokens": DEFAULT_GROQ_MAX_TOKENS,
            "stream": False,
        }
    ]
    assert output == "grounded [C1]"


def test_configured_model_and_parameters_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    generator = GroqTextGenerator(
        api_key="test-key",
        model="custom/model",
        temperature=0.2,
        max_tokens=321,
    )

    generator.generate("prompt")

    assert completions.calls[0]["model"] == "custom/model"
    assert completions.calls[0]["temperature"] == 0.2
    assert completions.calls[0]["max_tokens"] == 321


@pytest.mark.parametrize("content", [None, "", " \n"])
def test_empty_completion_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
) -> None:
    _, completions = install_fake_client(monkeypatch, content)
    generator = GroqTextGenerator(api_key="test-key")

    with pytest.raises(GenerationBackendError, match="empty completion"):
        generator.generate("prompt")

    assert len(completions.calls) == 1


def test_missing_completion_choice_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    completions.empty_choices = True

    with pytest.raises(GenerationBackendError, match="no completion choices"):
        GroqTextGenerator(api_key="test-key").generate("prompt")


def test_malformed_completion_choice_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    completions.malformed_choice = True

    with pytest.raises(GenerationBackendError, match="empty completion"):
        GroqTextGenerator(api_key="test-key").generate("prompt")


def test_provider_failure_is_chained_as_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    provider_error = APIConnectionError(request=httpx.Request("POST", "https://groq"))
    completions.error = provider_error

    with pytest.raises(GenerationBackendError, match="connection failed") as error:
        GroqTextGenerator(api_key="test-key").generate("prompt")

    assert error.value.__cause__ is provider_error


@pytest.mark.parametrize(
    ("error_type", "status_code", "message"),
    [
        (AuthenticationError, 401, "authentication failed"),
        (RateLimitError, 429, "rate limit exceeded"),
    ],
)
def test_provider_status_failures_have_clear_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_type: Any,
    status_code: int,
    message: str,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    request = httpx.Request("POST", "https://groq")
    response = httpx.Response(status_code, request=request)
    provider_error = error_type("provider failure", response=response, body=None)
    completions.error = provider_error

    with pytest.raises(GenerationBackendError, match=message) as error:
        GroqTextGenerator(api_key="test-key").generate("prompt")

    assert error.value.__cause__ is provider_error


def test_generic_api_failure_is_chained_as_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    request = httpx.Request("POST", "https://groq")
    provider_error = APIError("provider failure", request, body=None)
    completions.error = provider_error

    with pytest.raises(
        GenerationBackendError, match="generation request failed"
    ) as error:
        GroqTextGenerator(api_key="test-key").generate("prompt")

    assert error.value.__cause__ is provider_error


def test_missing_environment_key_fails_only_when_backend_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(GenerationBackendError, match="GROQ_API_KEY"):
        GroqTextGenerator()


def test_environment_key_is_loaded_on_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, _ = install_fake_client(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", " environment-key ")

    GroqTextGenerator()

    assert factory.calls == [{"api_key": "environment-key", "max_retries": 0}]


def test_repository_dotenv_key_is_loaded_on_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import config

    factory, _ = install_fake_client(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("GROQ_API_KEY=file-only-test-value\n", encoding="utf-8")
    monkeypatch.setattr(config, "_REPOSITORY_ENV_FILE", dotenv_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    GroqTextGenerator()

    assert factory.calls == [
        {"api_key": "file-only-test-value", "max_retries": 0}
    ]


def test_generate_answer_uses_backend_and_keeps_citation_validation_above_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch, "The value is 42 [C1].")
    backend = GroqTextGenerator(api_key="test-key")

    result = generate_answer("What is returned?", context_bundle(), backend)

    assert result.answer == "The value is 42 [C1]."
    assert result.citation_ids == ("C1",)
    assert len(completions.calls) == 1

    completions.content = "Fabricated evidence [C9]."
    assert backend.generate("plain prompt") == "Fabricated evidence [C9]."
    with pytest.raises(CitationValidationError):
        generate_answer("What is returned?", context_bundle(), backend)


def test_empty_context_bypasses_provider_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, completions = install_fake_client(monkeypatch)
    backend = GroqTextGenerator(api_key="test-key")

    result = generate_answer("Question", ContextBundle((), ""), backend)

    assert result.citation_ids == ()
    assert completions.calls == []
