"""Hosted text generation through the official Groq Python SDK."""

from groq import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from src.config import GroqSettings

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_TEMPERATURE = 0.1
DEFAULT_GROQ_MAX_TOKENS = 1024


class GenerationBackendError(RuntimeError):
    """Raised when hosted generation cannot produce usable answer text."""


class GroqTextGenerator:
    """Synchronous plain-text generator backed by Groq Chat Completions."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_GROQ_TEMPERATURE,
        max_tokens: int = DEFAULT_GROQ_MAX_TOKENS,
    ) -> None:
        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = GroqSettings.from_environment().api_key
        if not isinstance(resolved_api_key, str) or not resolved_api_key.strip():
            raise GenerationBackendError(
                "GROQ_API_KEY is required to construct GroqTextGenerator"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")

        self._client = Groq(api_key=resolved_api_key.strip(), max_retries=0)
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Return one non-streaming completion for an already constructed prompt."""
        try:
            completion = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=False,
            )
        except AuthenticationError as error:
            raise GenerationBackendError("Groq authentication failed") from error
        except RateLimitError as error:
            raise GenerationBackendError("Groq rate limit exceeded") from error
        except APIConnectionError as error:
            raise GenerationBackendError("Groq connection failed") from error
        except APIError as error:
            raise GenerationBackendError("Groq generation request failed") from error

        choices = getattr(completion, "choices", None)
        if not choices:
            raise GenerationBackendError("Groq returned no completion choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise GenerationBackendError("Groq returned an empty completion")
        return content
