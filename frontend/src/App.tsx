import { FormEvent, useState } from "react";

import {
  ApiError,
  AskResponse,
  RepositoryResponse,
  askQuestion,
  deleteRepository,
  loadRepository,
} from "./api";

type Activity = "idle" | "loading" | "asking" | "releasing";

function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [commit, setCommit] = useState("");
  const [repository, setRepository] = useState<RepositoryResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [activity, setActivity] = useState<Activity>("idle");
  const [error, setError] = useState<string | null>(null);

  const busy = activity !== "idle";

  async function handleLoad(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);
    setActivity("loading");
    try {
      const loaded = await loadRepository({
        repo_url: repoUrl.trim(),
        commit: commit.trim() || null,
      });
      setRepository(loaded);
    } catch (caught) {
      setError(readableError(caught, "The repository could not be prepared."));
    } finally {
      setActivity("idle");
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!repository) return;
    setError(null);
    setResult(null);
    setActivity("asking");
    try {
      setResult(await askQuestion(repository.repository_id, question.trim()));
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 404) {
        setRepository(null);
        setError(
          "This repository session is no longer available. Load the repository again.",
        );
      } else {
        setError(readableError(caught, "The question could not be answered."));
      }
    } finally {
      setActivity("idle");
    }
  }

  async function handleChangeRepository() {
    if (!repository) return;
    const repositoryId = repository.repository_id;
    setActivity("releasing");
    setError(null);
    try {
      await deleteRepository(repositoryId);
    } catch (caught) {
      if (!(caught instanceof ApiError && caught.status === 404)) {
        setError(
          readableError(caught, "The previous repository could not be released."),
        );
      }
    } finally {
      setRepository(null);
      setResult(null);
      setQuestion("");
      setActivity("idle");
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Codebase RAG home">
          <span className="brand-mark" aria-hidden="true">CR</span>
          <span>Codebase RAG</span>
        </a>
        <span className="header-label">Grounded code intelligence</span>
      </header>

      <main id="top" className="content">
        <section className="hero">
          <p className="eyebrow">Repository-aware answers</p>
          <h1>Understand a codebase without losing the source.</h1>
          <p className="hero-copy">
            Load a public GitHub repository, ask one focused question, and inspect the
            exact evidence behind every validated citation.
          </p>
        </section>

        <section className="workspace" aria-label="Codebase question answering workspace">
          <div className="step-heading">
            <span className="step-number">01</span>
            <div>
              <h2>Repository</h2>
              <p>Pin a public codebase to an immutable commit.</p>
            </div>
          </div>

          {!repository ? (
            <form className="repository-form" onSubmit={handleLoad}>
              <label>
                <span>GitHub repository URL</span>
                <input
                  type="url"
                  placeholder="https://github.com/owner/repository.git"
                  value={repoUrl}
                  onChange={(event) => setRepoUrl(event.target.value)}
                  disabled={busy}
                  required
                  autoComplete="url"
                />
              </label>
              <label>
                <span>Commit SHA <em>optional</em></span>
                <input
                  type="text"
                  placeholder="Use default branch HEAD"
                  value={commit}
                  onChange={(event) => setCommit(event.target.value)}
                  disabled={busy}
                  pattern="[0-9a-fA-F]{40}|[0-9a-fA-F]{64}"
                />
              </label>
              <button className="primary-button" type="submit" disabled={busy}>
                {activity === "loading" ? <SpinnerLabel text="Loading and indexing repository..." /> : "Load repository"}
              </button>
            </form>
          ) : (
            <RepositorySummary repository={repository} onChange={handleChangeRepository} busy={busy} />
          )}

          <div className="section-divider" />

          <div className="step-heading">
            <span className="step-number">02</span>
            <div>
              <h2>Question</h2>
              <p>Ask about behavior, relationships, configuration, or data flow.</p>
            </div>
          </div>

          <form className="question-form" onSubmit={handleAsk}>
            <label htmlFor="question">Ask a question about this codebase</label>
            <textarea
              id="question"
              rows={4}
              placeholder={repository ? "How does the main pipeline handle failures?" : "Load a repository to begin"}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={!repository || busy}
              required
            />
            <div className="question-actions">
              <span className="input-hint">Answers are grounded in retrieved source evidence.</span>
              <button
                className="primary-button ask-button"
                type="submit"
                disabled={!repository || busy || !question.trim()}
              >
                {activity === "asking" ? <SpinnerLabel text="Analyzing codebase..." /> : "Ask"}
              </button>
            </div>
          </form>
        </section>

        <div className="status-region" aria-live="polite">
          {error && <div className="error-banner" role="alert"><strong>Request failed</strong><span>{error}</span></div>}
          {result && <AnswerPanel result={result} />}
        </div>
      </main>

      <footer>
        <span>Codebase RAG demo</span>
        <span>Exact-search retrieval · validated citations</span>
      </footer>
    </div>
  );
}

function RepositorySummary({
  repository,
  onChange,
  busy,
}: {
  repository: RepositoryResponse;
  onChange: () => void;
  busy: boolean;
}) {
  return (
    <div className="repository-summary">
      <div className="ready-indicator"><span />Repository ready</div>
      <div className="repository-title">{repository.repo_url.replace(/\.git$/, "")}</div>
      <dl className="repository-meta">
        <div><dt>Commit</dt><dd title={repository.commit_sha}>{shortSha(repository.commit_sha)}</dd></div>
        <div><dt>Sources</dt><dd>{repository.source_file_count}</dd></div>
        <div><dt>Chunks</dt><dd>{repository.chunk_count}</dd></div>
        <div><dt>Dense index</dt><dd>{repository.dense_index_status}</dd></div>
      </dl>
      <button className="text-button" type="button" onClick={onChange} disabled={busy}>
        {busy ? "Releasing repository..." : "Change repository"}
      </button>
    </div>
  );
}

function AnswerPanel({ result }: { result: AskResponse }) {
  return (
    <section className="answer-section" aria-labelledby="answer-heading">
      <div className="answer-heading-row">
        <div>
          <p className="eyebrow">Grounded response</p>
          <h2 id="answer-heading">Answer</h2>
        </div>
        <span className="citation-count">{result.citations.length} cited {result.citations.length === 1 ? "source" : "sources"}</span>
      </div>
      <div className="answer-copy">{result.answer}</div>

      <div className="evidence-heading">
        <h3>Sources / Supporting Code</h3>
        <p>Each snippet is the actual repository chunk behind a validated citation.</p>
      </div>
      <div className="evidence-list">
        {result.citations.map((citation, index) => (
          <details className="evidence-card" key={citation.evidence_id} open={index === 0}>
            <summary>
              <span className="citation-chip">{citation.citation_id}</span>
              <span className="source-name">{citation.source}</span>
              <span className="line-label">{lineLabel(citation.start_line, citation.end_line)}</span>
              <span className={`origin-badge ${citation.origin}`}>{citation.origin === "relationship" ? "related" : "retrieved"}</span>
            </summary>
            <div className="evidence-detail">
              <div className="evidence-meta">
                <span>Chunk {citation.chunk_index}</span>
                <span>{citation.evidence_id}</span>
              </div>
              <pre><code>{citation.snippet}</code></pre>
            </div>
          </details>
        ))}
      </div>
    </section>
  );
}

function SpinnerLabel({ text }: { text: string }) {
  return <span className="spinner-label"><span className="spinner" aria-hidden="true" />{text}</span>;
}

function shortSha(sha: string): string {
  return sha.slice(0, 12);
}

function lineLabel(start: number | null, end: number | null): string {
  if (start === null || end === null) return "Lines unavailable";
  return start === end ? `Line ${start}` : `Lines ${start}–${end}`;
}

function readableError(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

export default App;
