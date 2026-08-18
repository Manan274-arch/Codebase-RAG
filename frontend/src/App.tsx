import { type FormEvent, type ReactNode, useState } from "react";

import {
  ApiError,
  AskResponse,
  CitationEvidence,
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
          <span className="brand-copy">
            <strong>Codebase RAG</strong>
            <small>Ask the source</small>
          </span>
        </a>
        <span className="header-label">
          <i aria-hidden="true" /> Grounded code intelligence
        </span>
      </header>

      <main id="top" className="content">
        <section className="hero">
          <div className="hero-message">
            <p className="eyebrow">Repository-aware answers</p>
            <h1>Follow the answer all the way back to the code.</h1>
            <p className="hero-copy">
              Bring a public repository, ask one focused question, and get a calm,
              readable explanation with the exact source evidence attached.
            </p>
          </div>
          <aside className="hero-proof" aria-label="How answers are prepared">
            <p className="mini-label">From question to proof</p>
            <ol>
              <li><span>01</span><div><strong>Find</strong><small>Lexical and semantic retrieval</small></div></li>
              <li><span>02</span><div><strong>Refine</strong><small>Reranking and linked context</small></div></li>
              <li><span>03</span><div><strong>Explain</strong><small>One grounded answer with citations</small></div></li>
            </ol>
          </aside>
        </section>

        <div className="workbench-grid">
          <section className="workspace" aria-label="Codebase question answering workspace">
            <div className="step-heading">
              <span className="step-number">01</span>
              <div>
                <h2>Choose a repository</h2>
                <p>We pin it to one immutable commit before reading the source.</p>
              </div>
            </div>

            {!repository ? (
              <form className="repository-form" onSubmit={handleLoad}>
                <label className="repository-url-field">
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
                    placeholder="Default branch HEAD"
                    value={commit}
                    onChange={(event) => setCommit(event.target.value)}
                    disabled={busy}
                    pattern="[0-9a-fA-F]{40}|[0-9a-fA-F]{64}"
                  />
                </label>
                <button
                  aria-label="Load repository"
                  className="primary-button"
                  type="submit"
                  disabled={busy}
                >
                  {activity === "loading" ? (
                    <SpinnerLabel text="Preparing repository..." />
                  ) : (
                    "Prepare repository"
                  )}
                </button>
              </form>
            ) : (
              <RepositorySummary
                repository={repository}
                onChange={handleChangeRepository}
                busy={busy}
              />
            )}

            <div className="section-divider" />

            <div className="step-heading">
              <span className="step-number">02</span>
              <div>
                <h2>Ask about the code</h2>
                <p>Behavior, relationships, configuration, data flow, or ownership.</p>
              </div>
            </div>

            <form className="question-form" onSubmit={handleAsk}>
              <label htmlFor="question">Your question</label>
              <textarea
                aria-label="Ask a question about this codebase"
                id="question"
                rows={5}
                placeholder={
                  repository
                    ? "How does the main pipeline handle failures?"
                    : "Prepare a repository to begin"
                }
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                disabled={!repository || busy}
                required
              />
              <div className="question-actions">
                <span className="input-hint">
                  Focused questions usually produce the clearest evidence trail.
                </span>
                <button
                  aria-label="Ask"
                  className="primary-button ask-button"
                  type="submit"
                  disabled={!repository || busy || !question.trim()}
                >
                  {activity === "asking" ? (
                    <SpinnerLabel text="Reading the code..." />
                  ) : (
                    <span className="button-label">Ask the repository <span aria-hidden="true">&rarr;</span></span>
                  )}
                </button>
              </div>
            </form>
          </section>

          <aside className="guide-panel" aria-label="Question guidance">
            <p className="mini-label">A useful starting point</p>
            <h2>Ask for a flow, not just a filename.</h2>
            <p>
              The retriever is strongest when it can connect responsibilities across
              functions and files.
            </p>
            <div className="prompt-example">
              <span>Try asking</span>
              <q>How does data move from the API route to persistence?</q>
            </div>
            <ul className="trust-list">
              <li><i aria-hidden="true" /> Public source is read, never executed</li>
              <li><i aria-hidden="true" /> Every displayed source is citation-validated</li>
              <li><i aria-hidden="true" /> Repository sessions can be released anytime</li>
            </ul>
          </aside>
        </div>

        <div className="status-region" aria-live="polite">
          {error && (
            <div className="error-banner" role="alert">
              <span className="error-mark" aria-hidden="true">!</span>
              <div><strong>Request failed</strong><span>{error}</span></div>
            </div>
          )}
          {result && <AnswerPanel result={result} />}
        </div>
      </main>

      <footer>
        <span>Codebase RAG demo</span>
        <span>Exact-search retrieval &middot; validated citations</span>
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
      <div className="repository-summary-top">
        <div>
          <div className="ready-indicator"><span />Repository ready</div>
          <div className="repository-title">
            {repository.repo_url.replace(/\.git$/, "")}
          </div>
        </div>
        <button className="text-button" type="button" onClick={onChange} disabled={busy}>
          {busy ? "Releasing..." : "Change repository"}
        </button>
      </div>
      <dl className="repository-meta">
        <div><dt>Commit</dt><dd title={repository.commit_sha}>{shortSha(repository.commit_sha)}</dd></div>
        <div><dt>Source files</dt><dd>{repository.source_file_count}</dd></div>
        <div><dt>Code chunks</dt><dd>{repository.chunk_count}</dd></div>
        <div><dt>Dense index</dt><dd>{repository.dense_index_status}</dd></div>
      </dl>
    </div>
  );
}

function AnswerPanel({ result }: { result: AskResponse }) {
  const validCitationIds = new Set(result.citation_ids);

  return (
    <section className="answer-section" aria-labelledby="answer-heading">
      <div className="answer-heading-row">
        <div>
          <p className="eyebrow">Grounded response</p>
          <h2 id="answer-heading">A clearer path through the code</h2>
        </div>
        <span className="citation-count">
          <i aria-hidden="true" />
          {result.citations.length} validated {result.citations.length === 1 ? "source" : "sources"}
        </span>
      </div>

      <div className="answer-question">
        <span>Question</span>
        <p>{result.question}</p>
      </div>

      <div className="answer-layout">
        <article className="answer-narrative" aria-label="Generated answer">
          <div className="answer-label"><span aria-hidden="true">A</span> Answer</div>
          <AnswerContent answer={result.answer} validCitationIds={validCitationIds} />
        </article>
        <aside className="answer-note">
          <span className="note-icon" aria-hidden="true">&#10003;</span>
          <strong>Evidence checked</strong>
          <p>Every citation shown below maps to a retrieved repository chunk.</p>
        </aside>
      </div>

      <div className="evidence-heading">
        <div>
          <p className="mini-label">Trace the response</p>
          <h3>Sources / Supporting Code</h3>
        </div>
        <p>Open a source to inspect the exact evidence supplied to the model.</p>
      </div>

      {result.citations.length > 0 ? (
        <div className="evidence-list">
          {result.citations.map((citation, index) => (
            <EvidenceCard citation={citation} index={index} key={citation.evidence_id} />
          ))}
        </div>
      ) : (
        <div className="empty-evidence">
          No source citations were included in this response.
        </div>
      )}
    </section>
  );
}

function AnswerContent({
  answer,
  validCitationIds,
}: {
  answer: string;
  validCitationIds: Set<string>;
}) {
  const blocks = answer.trim().split(/\n\s*\n/).filter(Boolean);

  return (
    <div className="answer-copy">
      {blocks.map((block, blockIndex) => {
        const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
        if (lines.length > 0 && lines.every((line) => /^[-*]\s+/.test(line))) {
          return (
            <ul key={blockIndex}>
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>
                  {renderCitationText(line.replace(/^[-*]\s+/, ""), validCitationIds)}
                </li>
              ))}
            </ul>
          );
        }
        if (lines.length > 0 && lines.every((line) => /^\d+\.\s+/.test(line))) {
          return (
            <ol key={blockIndex}>
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>
                  {renderCitationText(line.replace(/^\d+\.\s+/, ""), validCitationIds)}
                </li>
              ))}
            </ol>
          );
        }
        return <p key={blockIndex}>{renderCitationText(block, validCitationIds)}</p>;
      })}
    </div>
  );
}

function renderCitationText(text: string, validCitationIds: Set<string>): ReactNode[] {
  return text.split(/(\[C\d+\])/g).map((part, index) => {
    const citationId = /^\[(C\d+)\]$/.exec(part)?.[1];
    if (citationId && validCitationIds.has(citationId)) {
      return <span className="inline-citation" key={`${citationId}-${index}`}>{citationId}</span>;
    }
    return part;
  });
}

function EvidenceCard({ citation, index }: { citation: CitationEvidence; index: number }) {
  return (
    <details className="evidence-card" open={index === 0}>
      <summary>
        <span className="citation-chip">{citation.citation_id}</span>
        <span className="source-stack">
          <span className="source-name">{citation.source}</span>
          <span className="source-location">
            {lineLabel(citation.start_line, citation.end_line)} &middot; Chunk {citation.chunk_index}
          </span>
        </span>
        <span className={`origin-badge ${citation.origin}`}>
          {citation.origin === "relationship" ? "related" : "retrieved"}
        </span>
        <span className="disclosure-icon" aria-hidden="true" />
      </summary>
      <CodeSnippet citation={citation} />
    </details>
  );
}

function CodeSnippet({ citation }: { citation: CitationEvidence }) {
  const [copied, setCopied] = useState(false);

  async function copySnippet() {
    try {
      await navigator.clipboard.writeText(citation.snippet);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="evidence-detail">
      <div className="code-toolbar">
        <span className="window-dots" aria-hidden="true"><i /><i /><i /></span>
        <span className="evidence-id" title={citation.evidence_id}>{citation.evidence_id}</span>
        <button type="button" onClick={copySnippet} aria-label={`Copy ${citation.citation_id} evidence`}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre><code>{citation.snippet}</code></pre>
    </div>
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
  return start === end ? `Line ${start}` : `Lines ${start}-${end}`;
}

function readableError(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

export default App;
