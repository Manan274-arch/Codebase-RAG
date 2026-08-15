# Codebase RAG — Technical Architecture

## 1. Project Goal

This project is the retrieval and context foundation for a multi-language codebase
question-answering RAG system. Given a natural-language question about a repository, it
builds a structured code corpus, retrieves relevant code with directly linked context,
and packages that evidence deterministically.

The current implementation stops after context construction. LLM answers,
generated-answer citations, an application/API layer, and a frontend are not yet
implemented.

## 2. Current System Architecture

The system separates corpus construction, high-recall candidate generation, relevance
ranking, and deterministic context completion:

- Ingestion discovers supported source files and creates LangChain `Document` objects.
- Language-aware chunking produces stable, source-located code chunks.
- Structural analysis enriches chunks with definitions, imports, routes, and outbound
  HTTP calls.
- Relationship linking connects compatible frontend/client HTTP calls to backend routes.
- BM25 and dense retrieval independently contribute bounded candidate lists.
- Dense vectors live in a validated persistent local Qdrant collection.
- A score-free union deduplicates candidates by canonical chunk identity.
- A local cross-encoder determines relevance order.
- Bounded one-hop relationship expansion adds explicitly connected chunks.
- Context construction assigns stable evidence identities and query-local citation aliases.

The corpus representation contains a deterministic structural summary followed by the
original code. Exact source remains available in `metadata["raw_content"]`.

## 3. Ingestion and Corpus Construction

### Source discovery and loading

Repository discovery recursively finds supported source files while excluding common
version-control, dependency, cache, generated-output, and virtual-environment
directories. Paths are repository-relative and deterministic. Each source file is read
as UTF-8 into one LangChain `Document` with its source path and normalized language in
metadata.

### Language-aware chunking

LangChain recursive splitters create the shared chunk corpus. Supported languages use
language-specific splitting where LangChain provides it; other recognized languages use
the generic fallback. Each chunk retains its original metadata and receives a per-file
`chunk_index`, character offset, and inclusive source-line range.

The stable canonical identity used throughout retrieval is:

```text
source::chunk_index
```

### Structural enrichment

Tree-sitter-based analysis runs on whole-file documents independently of chunking. It
extracts lightweight syntax facts—including named definitions, imports, supported
backend routes, and supported outbound HTTP calls—with source spans. Those facts are
mapped onto chunks by line-range overlap and rendered into the shared retrieval text.

This is intentionally not compiler-grade semantic analysis. It does not provide global
symbol resolution, a complete call graph, or dependency-graph traversal.

### Relationship linking

Relationship linking compares statically recognized HTTP calls with backend routes by
HTTP method and normalized path segments. It supports conservative parameter matching
and deterministic specificity rules. Matching chunks receive reciprocal references in
plain metadata; source documents are not mutated.

The resulting links support bounded frontend-call ↔ backend-route context expansion.
They are not a general-purpose graph.

## 4. Retrieval Architecture Evolution

### 4.1 BM25 Baseline

BM25 established a transparent lexical baseline and remains useful when questions use
exact identifiers, paths, route names, or source terminology. It was insufficient alone
for paraphrased intent and relationship-oriented questions, where lexical overlap can be
weak or misleading.

The implementation uses `rank_bm25.BM25Okapi` with its default parameters (`k1=1.5`,
`b=0.75`, `epsilon=0.25`). The same regex tokenizer lowercases document and query text
and extracts ASCII alphanumeric/underscore runs. Punctuation, path separators, dots, and
route slashes are delimiters; underscores remain inside tokens; camelCase is not split;
and there is no stemming or stop-word removal. Corpus statistics are built once in the
retriever constructor. These frozen semantics were audited but not changed.

### 4.2 Dense Retrieval

Local code-aware embeddings added semantic retrieval and produced the largest improvement
over BM25. Dense retrieval reached candidate recall 1.0 by @50 on Benchmark v2, which
established it as the main semantic retrieval backbone.

### 4.3 BM25 + Dense with RRF

Reciprocal Rank Fusion was tested to combine complementary lexical and semantic ranks
without comparing incompatible scores. In the larger diagnostic benchmark, equal-weight
RRF often demoted strong dense candidates and did not improve the final system enough to
justify production use. RRF remains implemented as an evaluated baseline only.

### 4.4 Cross-Encoder Reranking

Candidate generation and final ranking were separated deliberately. First-stage
retrievers maximize the chance that useful code enters a bounded pool; the cross-encoder
then evaluates each query and candidate jointly. This substantially improved first-rank
and final ranking quality, so cross-encoder reranking is retained.

### 4.5 Relationship Expansion

Many code questions require more than one relevant chunk—for example, a frontend HTTP
call together with the backend route it invokes. Semantic ranking alone cannot guarantee
that both endpoints appear in the final context. Deterministic post-reranking expansion
raised linked-context coverage on eligible Benchmark v2 cases from about 0.3571 to 1.0,
so it is retained after reranking.

### 4.6 Final Candidate Union Experiment

The final experiment replaced competitive rank fusion with cooperative candidate
generation:

```text
BM25 top 25 + Dense top 25
        ↓
Score-Free Deduplicated Candidate Union
        ↓
Cross-Encoder
        ↓
Relationship Expansion
```

BM25 candidates are emitted first in branch order, followed by unseen dense candidates.
Duplicates are removed using `source::chunk_index`. Results preserve whether they came
from BM25, dense, or both, together with their real branch ranks. The union creates no
fused, normalized, averaged, or synthetic retrieval score; its pre-reranking order is
only a deterministic pooling policy.

On the architecture-diagnostic benchmark, union preserved Hit@1 and Hit@3 while slightly
improving deeper recall and graded ranking over dense-only and RRF candidate pipelines.
Branch overlap reduced the pool to roughly 38 unique candidates per query rather than
50, reducing cross-encoder work. This evidence selected union over RRF.

## 5. Persistent Index Migration

The original dense reference embedded the corpus into a normalized NumPy matrix and
computed exact cosine similarity against every chunk. The migration preserved the same
`jinaai/jina-embeddings-v2-base-code` model, 768-dimensional normalized document/query
vectors, enriched retrieval text, and cosine semantics.

`src/indexing/` now owns embedding and persistent-index lifecycle. Each chunk becomes a
Qdrant point with a deterministic UUID derived from `source::chunk_index`; its JSON-safe
payload reconstructs the original LangChain `Document`. Collection metadata records the
schema version, embedding model, and a deterministic corpus fingerprint. Matching indexes
are reused without re-embedding, while changed/incompatible corpora require an explicit
rebuild.

The old brute-force implementation remains under `src/evaluation/` as migration evidence,
not as a production backend. Persistent Qdrant exact search reproduced its top-25 results
and labeled metrics exactly, and the full pipeline preserved nDCG@5 0.8320 and linked
coverage 1.0.

This phase uses Qdrant's path-based local Python mode. Inspection of `qdrant-client 1.19`
and the collection evidence showed that local dense search performs a NumPy full scan;
`exact=False` and `hnsw_ef` do not activate a real HNSW graph. Therefore no production
HNSW latency claim is made. Exact Qdrant search is selected for the current corpus;
server-mode HNSW validation is deferred to a later deployment phase.

Local `exact=False` checks at `hnsw_ef` 64, 128, and 256 all had top-25 overlap 1.0
with exact search and nearly identical median latency (about 39 ms). These results only
confirm local full-scan equivalence; they are not ANN or server-performance evidence.

## 6. Final Retrieval Architecture

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
BM25 top 25   Persistent Qdrant
              Exact Dense top 25
      \          /
       Score-Free Deduplicated Candidate Union
                       ↓
                 Cross-Encoder
                       ↓
             Relationship Expansion
                       ↓
              Context Construction
                       ↓
          Citation-Aware Evidence Bundle
```

- BM25 is a complementary lexical candidate source.
- Qdrant exact dense retrieval is the primary semantic signal.
- RRF is retained only as an evaluated baseline.
- Candidate union assigns no combined retrieval score.
- The cross-encoder determines relevance ordering.
- Relationship expansion adds only explicitly linked context after reranking.
- Context construction packages final results without changing their retrieval order.

## 7. Context Construction

`src/generation/context.py` accepts the ordered `ExpandedSearchResult` sequence produced
after relationship expansion. It removes exact duplicate chunks by the existing
canonical identity while preserving the first occurrence and final result order.

Each `EvidenceItem` keeps two distinct identifiers:

```text
canonical evidence ID = source::chunk_index
citation alias = C1, C2, ...
```

The canonical ID is permanent corpus identity. The citation alias is assigned only
within one constructed context after deduplication. Evidence also retains source,
chunk index, unchanged page content, available source lines, and whether it originated
from normal retrieval or relationship expansion.

Rendered blocks use this deterministic format:

```text
[C1] src/api/users.py:42-67
<unchanged chunk page_content>

[C2] src/services/auth.py
<unchanged chunk page_content>
```

When source-line metadata is unavailable, no line range is invented. The optional
`max_chars` budget counts the exact rendered text, admits only complete prefix blocks,
and never truncates a chunk. No tokenizer or model-specific token policy is introduced
at this stage.

## 8. Evaluation Status

Retrieval is covered by deterministic frozen fixtures and offline evaluation at multiple
cutoffs using Hit Rate, Recall, MRR, graded nDCG, category metrics, and linked-context
coverage. Benchmark v2 contains development and held-out splits, but its results were
inspected repeatedly during architecture development. It is therefore an
**architecture-diagnostic benchmark**, not untouched external validation.

A separate untouched confirmation benchmark should precede strong external-quality
claims. Unless downstream evaluation exposes a concrete retrieval problem, the current
retrieval architecture is frozen.

Focused retrieval tests, the full test suite, Ruff, and strict mypy currently pass.

## 9. Current Project Boundary / Next Stage

Implemented through context construction:

- repository ingestion and multi-language chunking;
- structural enrichment and conservative relationship linking;
- BM25 and dense retrieval;
- persistent local Qdrant indexing and exact dense search;
- score-free candidate union;
- cross-encoder reranking;
- relationship expansion;
- deterministic citation-aware context construction with optional character budgeting;
- offline retrieval evaluation and baseline experiments.

Not yet implemented:

- model-specific token budgeting for generation;
- LLM answer generation;
- grounded source citations and snippets in generated answers;
- application/API layer;
- frontend.

The next project phase begins with answer generation over the deterministic evidence
bundle.
