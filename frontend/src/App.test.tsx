import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import * as api from "./api";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return {
    ...original,
    loadRepository: vi.fn(),
    askQuestion: vi.fn(),
    deleteRepository: vi.fn(),
  };
});

const repository: api.RepositoryResponse = {
  repository_id: "repository-1",
  repo_url: "https://github.com/example/project.git",
  commit_sha: "a".repeat(40),
  source_file_count: 12,
  chunk_count: 48,
  dense_index_status: "built",
};

const answer: api.AskResponse = {
  question: "Where is authentication handled?",
  answer: "Authentication is handled by authenticate [C1].",
  citation_ids: ["C1"],
  citations: [
    {
      citation_id: "C1",
      evidence_id: "src/auth.py::2",
      source: "src/auth.py",
      chunk_index: 2,
      snippet: "def authenticate(token: str) -> bool:\n    return bool(token)",
      start_line: 10,
      end_line: 11,
      origin: "retrieved",
    },
  ],
};

describe("Codebase RAG demo", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.mocked(api.loadRepository).mockReset();
    vi.mocked(api.askQuestion).mockReset();
    vi.mocked(api.deleteRepository).mockReset();
  });

  it("loads a repository and enables questions", async () => {
    const user = userEvent.setup();
    vi.mocked(api.loadRepository).mockResolvedValue(repository);
    render(<App />);

    const question = screen.getByLabelText("Ask a question about this codebase");
    expect(question).toBeDisabled();
    await user.type(
      screen.getByLabelText("GitHub repository URL"),
      repository.repo_url,
    );
    await user.click(screen.getByRole("button", { name: "Load repository" }));

    expect(await screen.findByText("Repository ready")).toBeInTheDocument();
    expect(question).toBeEnabled();
    expect(api.loadRepository).toHaveBeenCalledWith({
      repo_url: repository.repo_url,
      commit: null,
    });
  });

  it("renders a generated answer and its citation evidence", async () => {
    const user = userEvent.setup();
    vi.mocked(api.loadRepository).mockResolvedValue(repository);
    vi.mocked(api.askQuestion).mockResolvedValue(answer);
    render(<App />);

    await user.type(screen.getByLabelText("GitHub repository URL"), repository.repo_url);
    await user.click(screen.getByRole("button", { name: "Load repository" }));
    await user.type(
      screen.getByLabelText("Ask a question about this codebase"),
      answer.question,
    );
    await user.click(screen.getByRole("button", { name: "Ask" }));

    const generatedAnswer = await screen.findByRole("article", {
      name: "Generated answer",
    });
    expect(generatedAnswer).toHaveTextContent(
      "Authentication is handled by authenticate",
    );
    expect(generatedAnswer).toHaveTextContent("C1");
    expect(screen.getByText("Sources / Supporting Code")).toBeInTheDocument();
    expect(screen.getAllByText("C1")).toHaveLength(2);
    expect(screen.getByText("src/auth.py")).toBeInTheDocument();
    expect(screen.getByText(/Lines 10/)).toBeInTheDocument();
    expect(screen.getByText(/def authenticate\(token: str\)/)).toBeInTheDocument();
  });

  it("shows a useful backend failure without rendering raw response data", async () => {
    const user = userEvent.setup();
    vi.mocked(api.loadRepository).mockRejectedValue(
      new api.ApiError("repository acquisition failed", 400),
    );
    render(<App />);

    await user.type(
      screen.getByLabelText("GitHub repository URL"),
      repository.repo_url,
    );
    await user.click(screen.getByRole("button", { name: "Load repository" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "repository acquisition failed",
    );
  });
});
