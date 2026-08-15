# Multi-Language Codebase Q&A RAG Assistant

This project is the retrieval and context foundation for a codebase question-answering
RAG system. It ingests a software repository, retrieves relevant code plus explicitly
linked context, and packages that evidence for natural-language questions.

The repository currently implements ingestion through deterministic context
construction. Answer generation, generated-answer citations, an API/application layer,
and a frontend are not yet built.

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
answer generation are not implemented yet.

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
- offline retrieval evaluation.

Next:

- model-specific token budgeting;
- answer generation;
- grounded citations and source snippets;
- API/application layer;
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
