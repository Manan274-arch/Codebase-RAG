export interface RepositoryRequest {
  repo_url: string;
  commit: string | null;
}

export interface RepositoryResponse {
  repository_id: string;
  repo_url: string;
  commit_sha: string;
  source_file_count: number;
  chunk_count: number;
  dense_index_status: "built" | "reused";
}

export interface CitationEvidence {
  citation_id: string;
  evidence_id: string;
  source: string;
  chunk_index: number;
  snippet: string;
  start_line: number | null;
  end_line: number | null;
  origin: "retrieved" | "relationship";
}

export interface AskResponse {
  question: string;
  answer: string;
  citation_ids: string[];
  citations: CitationEvidence[];
}

interface ErrorPayload {
  detail?: string | Array<{ msg?: string }>;
}

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_BASE_URL = configuredBaseUrl.replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });
  } catch {
    throw new ApiError(
      "The backend is unavailable. Confirm that the API is running on port 8000.",
      null,
    );
  }

  if (!response.ok) {
    const payload = await safeErrorPayload(response);
    throw new ApiError(errorMessage(payload, response.status), response.status);
  }
  return (await response.json()) as T;
}

async function safeErrorPayload(response: Response): Promise<ErrorPayload> {
  try {
    return (await response.json()) as ErrorPayload;
  } catch {
    return {};
  }
}

function errorMessage(payload: ErrorPayload, status: number): string {
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    const messages = payload.detail
      .map((item) => item.msg)
      .filter((message): message is string => Boolean(message));
    if (messages.length > 0) {
      return messages.join(" ");
    }
  }
  return `The API request failed with status ${status}.`;
}

export function loadRepository(payload: RepositoryRequest): Promise<RepositoryResponse> {
  return request<RepositoryResponse>("/api/repositories", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function askQuestion(
  repositoryId: string,
  question: string,
): Promise<AskResponse> {
  return request<AskResponse>(`/api/repositories/${repositoryId}/ask`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function deleteRepository(repositoryId: string): Promise<void> {
  return request(`/api/repositories/${repositoryId}`, {
    method: "DELETE",
  });
}
