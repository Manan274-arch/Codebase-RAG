# Multi-Language Codebase Q&A RAG Assistant

This project is the retrieval, context, and generation foundation for a codebase
question-answering RAG system. It ingests a software repository, retrieves relevant code
plus explicitly linked context, and generates citation-aware answers from that evidence.

The repository currently implements provider-agnostic generation orchestration,
deterministic citation validation, hosted inference through Groq, and a small FastAPI
demo backend. A frontend is not yet built.

## Architecture

```text
Repository
    ↓
Source Discovery
    ↓
LangChain Documents
    ↓
Language-Aware Chunking
    ↓
Structural Enrichment
    ↓
Relationship Linking
    ↓
Shared Code Corpus
      /          \
   BM25       Persistent Qdrant
 top 25      Exact Dense top 25
      \          /
    ↓
Score-Free Deduplicated Candidate Union
    ↓
Cross-Encoder Reranking
    ↓
Relationship Expansion
    ↓
Context Construction
Citation-Aware Evidence Bundle
    ↓
Citation-Aware Generation
    ↓
Groq API · openai/gpt-oss-20b
```

BM25 contributes lexical matches and dense retrieval supplies the primary semantic
signal. RRF was evaluated but is retained only as a benchmark baseline—not as part of
the selected production path.

## Current capabilities

- Deterministic repository source-file discovery.
- Multi-language UTF-8 source ingestion into LangChain `Document` objects.
- Language-aware chunking with stable source paths, chunk indices, offsets, and lines.
- Tree-sitter structural extraction for definitions and imports.
- Conservative extraction of supported backend routes and outbound HTTP calls.
- Lightweight HTTP-call ↔ backend-route relationship linking.
- A shared retrieval representation containing structural summaries and original code.
- Local BM25 and code-aware dense retrieval.
- Persistent local Qdrant vector indexing with corpus/configuration fingerprints.
- Score-free, provenance-preserving candidate union.
- Local cross-encoder reranking.
- Bounded one-hop relationship expansion after reranking.
- Deterministic context construction with stable evidence IDs and query-local citations.
- Provider-agnostic grounded generation with deterministic citation validation.
- Hosted Groq generation using `openai/gpt-oss-20b` by default.
- A process-local FastAPI backend that retains prepared repository sessions.
- Offline retrieval evaluation with frozen fixtures and graded relevance labels.

## Retrieval design

BM25 and Qdrant dense retrieval independently return their top 25 candidates. BM25
builds its lexical statistics in memory when the corpus is loaded. Dense document
embeddings are built separately and persisted in local Qdrant storage; query-time dense
retrieval embeds only the query and searches the validated collection using exact cosine
search. The
`CandidateUnionRetriever` combines them deterministically, removes duplicate
`source::chunk_index` identities, and records whether each candidate came from BM25,
dense, or both, including its original branch ranks. It does not create a fused score.

The cross-encoder jointly scores the question and every unique candidate to determine
the relevance order. Relationship expansion then inserts bounded, directly linked
chunks such as the backend route associated with a retrieved frontend HTTP call.

Context construction consumes that final ordered result list, deduplicates it by the
existing canonical identity, and renders complete evidence blocks without modifying
chunk content. A permanent canonical evidence ID uses `source::chunk_index`; the
separate query-local citation alias uses `C1`, `C2`, and so on. An optional character
budget admits only complete, rank-preserving blocks. Model-specific token budgeting and
end-to-end generated-answer evaluation are not implemented yet.

Generation consumes the existing `ContextBundle` without rebuilding its evidence text.
An injected text generator receives a concise grounding prompt, and generated citation
aliases such as `[C1]` are checked against the bundle. Unknown aliases are rejected
rather than removed or accepted. The bundle retains the mapping from each query-local
alias to its permanent `source::chunk_index` evidence identity.

`TextGenerator` remains the provider boundary. The first concrete implementation,
`GroqTextGenerator`, sends only the constructed prompt to Groq's hosted Chat Completions
API and defaults to `openai/gpt-oss-20b`. Retrieval and context construction contain no
Groq-specific logic, so another backend or model can be substituted without redesigning
the pipeline.

On Benchmark v2, branch overlap leaves roughly 38 unique candidates per query on
average instead of the maximum 50.

Persistent local mode was selected to keep the existing Python workflow. A real HNSW
path cannot be validated in this mode: `qdrant-client` performs a NumPy full scan and
ignores HNSW search parameters.
HNSW remains a later server/deployment concern rather than a claimed local benchmark.

## Evaluation snapshot

Final held-out Benchmark v2 results after cross-encoding and relationship expansion:

| Candidate strategy | Hit@1 | Hit@3 | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.8857 | 0.9571 | 0.8762 | 0.8255 |
| RRF baseline | 0.8857 | 0.9571 | 0.8762 | 0.8255 |
| Candidate union | 0.8857 | 0.9571 | 0.8881 | 0.8320 |

Union was selected because it preserved early ranking quality, slightly improved deeper
recall and graded ranking, produced no measured per-query nDCG@5 regression, and
required fewer cross-encoder candidate evaluations after deduplication. Relationship
expansion achieved Linked Context Coverage@5 of 1.0 on all eligible held-out cases.

Benchmark v2 has been inspected during architecture development and is therefore an
architecture-diagnostic benchmark, not untouched external validation.

Migration validation found identical brute-force and Qdrant-exact candidate metrics,
top-25 identity overlap of 1.0, and unchanged final nDCG@5 of 0.8320. With only 56
benchmark chunks, measured exact-search latency was similar to the old NumPy reference;
Qdrant was selected for persistence and index lifecycle rather than a small-corpus speed
claim.

Reproduce the final candidate-strategy comparison with:

```shell
python -m src.evaluation.runners.evaluate_qdrant_migration
```

The first real-model run may download `jinaai/jina-embeddings-v2-base-code` and
`cross-encoder/ms-marco-MiniLM-L6-v2`; subsequent runs use the local model cache.

## Project status

Completed:

- ingestion and source discovery;
- chunking and corpus construction;
- structural enrichment;
- relationship linking;
- lexical and dense retrieval;
- persistent Qdrant indexing and exact dense search;
- hybrid/RRF retrieval experiments;
- candidate-union experiments;
- cross-encoder reranking;
- relationship expansion;
- deterministic citation-aware context construction;
- provider-agnostic answer generation and citation validation;
- hosted Groq generation with `openai/gpt-oss-20b`;
- automatic Git acquisition and reusable end-to-end orchestration;
- process-local FastAPI demo backend;
- offline retrieval evaluation.

Next:

- model-specific token budgeting;
- end-to-end generated-answer evaluation;
- frontend.

## Development and testing

Python 3.11 is required.

```shell
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Place the Groq credential in the repository-root `.env`:

```text
GROQ_API_KEY=<your key>
```

`GroqTextGenerator()` loads this file automatically when constructed; no manual shell
environment setup is required. Existing process environment values take precedence.
Credentials are never required for ingestion, retrieval, context construction, or
offline tests. The local `.env` remains ignored by Git.

Run the manual live generation smoke test with:

```shell
python -m scripts.smoke_groq_generation
```

This sends one small synthetic `ContextBundle` through the real Groq backend and the
normal citation-validation path. It is a connectivity and contract check, not
end-to-end retrieval or generated-answer quality evaluation.

Run the complete RAG flow directly from a public Git repository URL:

```shell
python -m scripts.run_codebase_rag --repo https://github.com/OWNER/REPOSITORY.git --question "How does the main pipeline work?"
```

Repeat `--question` to ask several questions while reusing the same prepared corpus,
BM25 index, Qdrant collection, embedding model, and reranker. Use `--commit` with a
full commit SHA to select an exact revision. Without it, the command resolves the
remote default branch once and pins that SHA for the run. The URL, resolved SHA,
checkout path, source/chunk counts, index status, answer, validated citation IDs, and
cited evidence are printed.

Repositories are cached outside tracked source code under
`.cache/codebase-rag/repositories/`. Each immutable checkout is keyed by repository URL
and commit SHA, and clean cached checkouts are reused. Acquisition supports public
HTTPS Git URLs; private-repository authentication is intentionally out of scope.
Fetched repositories are untrusted input: the application only reads supported source
files through the existing ingestion pipeline. It never executes repository code,
runs setup scripts, or installs repository dependencies.

The full live smoke test used this exact pinned command (PowerShell backticks only wrap
the command for readability):

```powershell
python -m scripts.run_codebase_rag `
  --repo "https://github.com/Manan274-arch/Title-block-and-BOM-extraction-for-deciding-raw-materials.git" `
  --commit de4a29e4d453d0695b1a2cf5d76f8285032bea70 `
  --question "What stages does DrawingPipeline.run execute, and when does it stop early?" `
  --question "How does BOMExtractor identify BOM columns and decide which table rows are valid?" `
  --question "How does MaterialAggregator decide whether to derive raw materials from BOM rows or the title block, and how are quantities combined?" `
  --question "How is application configuration organized, and how does with_output_directory change output paths?"
```

Groq's token-per-minute limit may require running the same command with one question at
a time. The repository checkout and Qdrant collection are reused across those runs.

The same lifecycle is available from Python:

```python
from src.codebase_rag import CodebaseRAG

with CodebaseRAG.from_repository_url("https://github.com/OWNER/REPOSITORY.git") as rag:
    result = rag.ask("How does the main pipeline work?")
    print(result.answer)
    print(result.citation_ids)
```

### Demo backend

Start the Phase 3 FastAPI backend locally with:

```shell
uvicorn src.api.app:app --reload
```

It exposes only:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Confirm the process is alive. |
| `POST` | `/api/repositories` | Acquire and prepare a repository once. |
| `POST` | `/api/repositories/{repository_id}/ask` | Ask the prepared repository a question. |
| `DELETE` | `/api/repositories/{repository_id}` | Close and remove the repository session. |

Repository IDs are opaque and sessions exist only in this Python process. Deleting a
session calls `CodebaseRAG.close()`, and application shutdown closes every retained
session. This demo intentionally has no database, cross-worker synchronization,
authentication, background queue, or deployment layer. The direct Python API above
remains supported and unchanged.

Build or validate a persistent local index for a repository:

```shell
python -m src.indexing.build_index PATH_TO_REPOSITORY
```

The default store is `.qdrant/` and is ignored by Git. A matching corpus and embedding
configuration reuses the collection without re-embedding. A changed or incompatible
corpus requires an explicit rebuild:

```shell
python -m src.indexing.build_index PATH_TO_REPOSITORY --rebuild
```

`CODEBASE_RAG_QDRANT_PATH` and `CODEBASE_RAG_QDRANT_COLLECTION` can override the local
path and collection name. Exact search is the default.

Run the quality gates:

```shell
pytest
ruff check .
mypy src
```

Additional historical evaluation entry points remain available under
`src/evaluation/runners/`. They reproduce evaluated baselines and do not define the
production retrieval architecture.

For architectural rationale and project boundaries, see [DOCUMENT.md](DOCUMENT.md).
