# Codebase RAG

A multi-language codebase question-answering system that accepts a public Git
repository, indexes its source code, retrieves relevant chunks, and generates a
grounded answer with validated citations.

## How it works

```text
Git URL + optional commit SHA
        -> immutable cached checkout
        -> source discovery and language-aware chunking
        -> structural and HTTP-relationship enrichment
        -> BM25 top 25 + Qdrant exact dense top 25
        -> deduplicated candidate union
        -> cross-encoder reranking
        -> bounded relationship expansion
        -> citation-labeled context (up to 10 chunks by default)
        -> one Groq generation call
        -> citation validation
        -> answer with supporting source snippets
```

The production retrieval path uses lexical and semantic search cooperatively. BM25
finds exact identifiers and code terms; Qdrant finds semantic matches using
`jinaai/jina-embeddings-v2-base-code`. The union creates no synthetic score. A
`cross-encoder/ms-marco-MiniLM-L6-v2` model determines the final relevance order,
after which explicit frontend HTTP-call/backend-route relationships may add supporting
chunks within the final result limit.

Groq receives only the final context and question, not the repository. Citation aliases
such as `[C1]` are assigned per question and validated before evidence is returned to
the client.

## Results

### Retrieval

The frozen Retrieval Benchmark v2 held-out split contains 70 graded queries over a
56-chunk corpus. It is an architecture-diagnostic benchmark used during development,
not untouched external validation.

Final metrics below are measured after cross-encoder reranking and relationship
expansion.

| Candidate strategy | Hit@1 | Hit@3 | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense candidates | 88.57% | 95.71% | 87.62% | 82.55% |
| RRF candidates | 88.57% | 95.71% | 87.62% | 82.55% |
| **BM25 + dense candidate union (selected)** | **88.57%** | **95.71%** | **88.81%** | **83.20%** |

| Additional retrieval check | Result |
| --- | ---: |
| Linked Context Coverage@5 on eligible held-out queries | 100.00% |
| Qdrant exact vs. brute-force dense top-25 overlap | 100.00% |
| Final nDCG@5 after Qdrant migration | 83.20% |
| Average unique union candidates from a maximum of 50 | ~38 |

The selected union preserved early hit rate while improving deeper recall and graded
ranking. Persistent Qdrant exact search reproduced the old brute-force dense results;
Qdrant was selected for persistence and index lifecycle, not for a small-corpus latency
claim. Embedded local Qdrant performs exact full scans, so this repository does not
claim production HNSW performance.

Reproduce the retrieval comparisons with:

```shell
python -m evaluation.runners.evaluate_candidate_strategies
python -m evaluation.runners.evaluate_qdrant_migration
```

### Answer generation

The saved end-to-end comparison uses 18 manually curated questions against this public
repository and pinned commit:

```text
Repository: Manan274-arch/Title-block-and-BOM-extraction-for-deciding-raw-materials
Commit: de4a29e4d453d0695b1a2cf5d76f8285032bea70
Benchmark: title-block-bom-e2e-v1
```

Both systems used the same embedding model, context construction, Groq model,
temperature, and answer-token limit. Dense-only sent exact dense top-10 directly to
context. Final RAG used BM25+dense union, cross-encoder reranking, and relationship
expansion.

| Metric | Dense-only | Final RAG | Difference |
| --- | ---: | ---: | ---: |
| Citation precision | 58.33% | **63.98%** | **+5.65 pp** |
| Citation recall | 47.96% | **64.07%** | **+16.11 pp** |
| Groundedness | 86.11% | **87.50%** | **+1.39 pp** |
| Supporting-code accuracy | **88.89%** | 87.50% | -1.39 pp |
| Unanswerable accuracy | 100.00% | 100.00% | 0.00 pp |

The clearest gain was evidence completeness: cross-file citation recall increased from
36.67% to 66.67%, while cross-file citation precision increased from 56.67% to 73.33%.
Groundedness and supporting-code accuracy are rubric-judge measurements; citation
precision and recall are deterministic comparisons against expected supporting files.

These values come from the saved evaluation artifact. The latest prompt-only citation
instruction change has not been re-evaluated, so no improvement from that change is
claimed here.

Detailed answers, citations, retrieved chunks, context, judge records, and category
breakdowns are available in
[`evaluation/results/title_block_bom_de4a29e4/`](evaluation/results/title_block_bom_de4a29e4/).

## Main capabilities

- Public HTTPS Git acquisition with exact commit pinning and cached immutable checkouts.
- Multi-language source discovery and UTF-8 loading.
- Language-aware chunks with stable file, chunk, and line identities.
- Tree-sitter definitions/imports plus conservative route and HTTP-call extraction.
- Persistent local Qdrant indexes with corpus fingerprints and safe reuse.
- In-memory BM25, exact dense retrieval, candidate union, and cross-encoder reranking.
- Bounded one-hop relationship expansion.
- Deterministic context construction and citation validation.
- One-call Groq generation using `openai/gpt-oss-20b` by default.
- FastAPI backend and React/TypeScript demo.

Fetched repositories are treated as untrusted input. The application reads supported
source files but never executes repository code, runs setup scripts, or installs the
repository's dependencies.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/ingestion/` | Git acquisition, source discovery, and loading |
| `src/chunking/` | Language-aware chunking and source lines |
| `src/enrichment/` | Structure, routes, HTTP calls, relationships, retrieval text |
| `src/indexing/` | Embeddings and persistent Qdrant lifecycle |
| `src/retrieval/` | BM25, dense search, union, reranking, expansion |
| `src/generation/` | Context, prompting, Groq, citation validation |
| `src/pipeline/` | Corpus and end-to-end production orchestration |
| `backend/` | FastAPI request models and repository sessions |
| `frontend/` | React/Vite demo |
| `evaluation/` | Frozen benchmarks, historical baselines, and saved results |
| `tests/` | Unit and integration tests |

## Setup

Python 3.11 is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a repository-root `.env` file:

```env
GROQ_API_KEY=<your key>
```

The file is ignored by Git. Retrieval and offline tests do not require a Groq key.

## Run the complete pipeline

From the command line:

```powershell
python -m scripts.run_codebase_rag `
  --repo "https://github.com/OWNER/REPOSITORY.git" `
  --question "How does the main pipeline work?"
```

Add `--commit <FULL_SHA>` to evaluate an exact revision. Repeat `--question` to reuse
the prepared corpus and indexes for several questions.

From Python:

```python
from src.pipeline.codebase_rag import CodebaseRAG

with CodebaseRAG.from_repository_url(
    "https://github.com/OWNER/REPOSITORY.git"
) as rag:
    result = rag.ask("How does the main pipeline work?")
    print(result.answer)
    print(result.citation_ids)
```

Repositories are cached under `.cache/codebase-rag/repositories/`; Qdrant data is
stored under `.qdrant/`. Both paths are ignored by Git.

## Run the demo

Start the backend from the repository root:

```shell
uvicorn backend.app:app --reload
```

Start the frontend in a second terminal:

```shell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend uses `http://localhost:8000` by default.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/repositories` | Acquire and prepare a repository |
| `POST` | `/api/repositories/{id}/ask` | Ask a question |
| `DELETE` | `/api/repositories/{id}` | Release the repository session |

Repository sessions are process-local; this demo intentionally has no authentication,
database, background queue, or deployment layer.

## Quality checks

```shell
pytest
ruff check .
mypy
```

Frontend checks:

```shell
cd frontend
npm run typecheck
npm run test
npm run build
```

For detailed design decisions and stage-by-stage internals, see
[`DOCUMENT.md`](DOCUMENT.md).
