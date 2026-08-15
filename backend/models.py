"""HTTP request and response models for the demo API."""

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")


class ApiModel(BaseModel):
    """Reject accidental fields at the small public HTTP boundary."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"


class RepositoryRequest(ApiModel):
    repo_url: str = Field(min_length=1)
    commit: str | None = None

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("repo_url must be a public HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("repo_url must not contain credentials")
        return value

    @field_validator("commit")
    @classmethod
    def validate_commit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("commit must be a full hexadecimal Git commit SHA")
        return value.lower()


class RepositoryResponse(ApiModel):
    repository_id: str
    repo_url: str
    commit_sha: str
    source_file_count: int
    chunk_count: int
    dense_index_status: Literal["built", "reused"]


class AskRequest(ApiModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class CitationResponse(ApiModel):
    citation_id: str
    evidence_id: str
    source: str
    chunk_index: int
    snippet: str
    start_line: int | None
    end_line: int | None
    origin: Literal["retrieved", "relationship"]


class AskResponse(ApiModel):
    question: str
    answer: str
    citation_ids: list[str]
    citations: list[CitationResponse]


class DeleteResponse(ApiModel):
    status: Literal["deleted"] = "deleted"
    repository_id: str
