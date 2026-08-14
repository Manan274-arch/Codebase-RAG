# Technical Architecture

## Project objective

Build a multi-language Codebase Q&A RAG Assistant capable of answering
natural-language questions about a software repository using retrieved source-code
context.

## High-level pipeline

```text
repository
-> source-file discovery
-> LangChain Documents
-> language-aware or fallback chunking
-> code chunks with source ranges
-> line-range structural enrichment
-> enriched chunk corpus
```

In parallel with chunk creation, each whole-file Document now follows this structural
path:

```text
whole-file Document
-> Tree-sitter structural analysis
-> definitions, imports, backend routes, and outbound HTTP calls with source spans
```

These paths join deterministically:

```text
code chunks with source ranges + file structure
-> inclusive line-range overlap
-> structurally enriched code chunks
```

The remaining planned retrieval and generation pipeline is:

```text
             chunk corpus
             /          \
           BM25         dense
             \          /
        Reciprocal Rank Fusion
                  |
          cross-encoder reranking
                  |
          LLM answer generation
```

## Architectural principles

- Prefer LangChain for generic RAG infrastructure.
- Keep custom code focused on application-specific behavior.
- Support multiple programming languages rather than designing around Python ASTs.
- Avoid unnecessary abstractions.
- Build and test the system incrementally.
- Introduce dependencies only when a concrete feature needs them.

## Intended source responsibilities

- `src/ingestion/`: Repository discovery, source loading, language handling, and chunk
  preparation.
- `src/indexing/`: Embeddings and vector-store indexing.
- `src/retrieval/`: Retrieval and later optional reranking or query transformation.
- `src/generation/`: Prompt construction and LLM-based answer generation.
- `src/config.py`: Application configuration as actual configuration needs appear.

## Current implementation status

Repository source-file discovery, LangChain Document creation, and chunk-corpus
creation are implemented in `src/ingestion/`. Discovery recursively identifies
candidate source files while pruning version-control metadata, virtual environments,
dependency/vendor directories, generated build output, caches, coverage output, and
editor metadata. Symbolic-link directories are not traversed.

Discovery currently recognizes these extensions using case-insensitive matching:

```text
.py
.java
.c .h
.cc .cpp .cxx .hh .hpp .hxx
.js .jsx .mjs .cjs
.ts .tsx
.go
.rs
.cs
.rb
.php
.swift
.kt .kts
.scala
.sh .bash .zsh
.sql
```

Results are repository-relative paths sorted deterministically by their POSIX-style
representation. Language classification relies only on file extensions. `.h` files
are initially classified as C despite their ambiguity between C and C++; no parsing or
repository heuristics are used to resolve that ambiguity.

Each discovered file is read in full as UTF-8 and becomes exactly one LangChain
`Document`. Invalid UTF-8 raises `UnicodeDecodeError` rather than silently changing
source text. The current Document contract is:

- `page_content`: The entire source-file contents, exactly as decoded.
- `metadata["source"]`: A stable, repository-relative POSIX path string.
- `metadata["language"]`: A normalized, extension-derived language label.

Whole-file Documents are split into the future shared retrieval corpus with default
`chunk_size=1500` and `chunk_overlap=200`. Python, Java, C, C++, JavaScript,
TypeScript, Go, Rust, C#, Ruby, PHP, Swift, Kotlin, and Scala use LangChain's
language-aware recursive splitting. Languages without a corresponding LangChain
`Language` member, currently Shell and SQL, use generic recursive splitting and are
not discarded. No custom AST is used for chunking. Tree-sitter analysis is a separate
whole-file operation and does not replace LangChain's splitting behavior.

Every chunk preserves all parent metadata and adds:

- `chunk_index`: Zero-based within its source Document and restarted for each file.
- `start_index`: Character offset of the chunk in the complete source text, supplied
  by LangChain's public text-splitter API.
- `start_line` and `end_line`: One-based, inclusive source-line range derived from the
  exact character offset and unchanged chunk content.

Empty source Documents produce no chunks. Non-empty chunks contain source text only;
file paths and language labels remain metadata. Splitter whitespace stripping is
disabled so source whitespace is preserved. Parent and within-parent ordering are
preserved, making the output a deterministic shared corpus for future sparse and dense
retrieval.

## Structural extraction

Source discovery produces the supported code-file set, and the loader creates one
whole-file LangChain Document per source file. Language-aware chunking continues to
create code chunks independently. The structural layer analyzes the complete Document
text with `tree-sitter-language-pack` and produces lightweight definitions, imports,
backend route definitions, and outbound HTTP calls. It does not reread files or use
chunks as parser input.

The public API is `src.ingestion.structure.extract_structure(document)`. It returns
an immutable `FileStructure` containing the normalized language, optional source
identifier, and tuples of project-owned `Definition`, `Import`, `RouteDefinition`, and
`HttpCall` values. Vendor Tree-sitter objects do not cross this boundary. Nested definitions are
recursively flattened in source order; lexical ancestry is retained in names such as
`Service.get_user`. This is syntactic nesting, not symbol resolution.

Structural spans use one-based, inclusive line numbers. Columns are zero-based, with
an exclusive end column. The adapter performs the line conversion from Tree-sitter's
zero-based values.

The dependency's extraction capabilities vary by grammar. In the bounded 1.x API,
Python, JavaScript, and Java expose representative named definitions, nested methods,
and imports. Its current C/C++ structural output can report function/class spans
without names and does not classify preprocessor `#include` directives as imports.
The adapter omits unnamed entries rather than inventing identifiers.

## Backend route extraction

Route recognition performs one additional in-memory Tree-sitter parse of the already
loaded whole-file text. The high-level definitions/imports API does not expose its
syntax tree, so the proven Prompt 5 adapter remains unchanged. Recognizers traverse
confirmed decorator, annotation, call-expression, and method-declaration nodes; they
do not scan whole files with regular expressions.

The intentionally small support matrix is:

| Language | Framework | Recognized static forms |
| --- | --- | --- |
| Python | FastAPI | `app`/`router` HTTP-method decorators |
| Python | Flask | `app`/`blueprint` `route()` and HTTP-method decorators |
| JavaScript/TypeScript | Express | Direct `app`/`router.HTTP_METHOD(path, handler)` registrations |
| Java | Spring Web/MVC | Method `*Mapping`/`RequestMapping` annotations and lexical class prefixes |

Supported HTTP methods are GET, POST, PUT, PATCH, DELETE, OPTIONS, and HEAD. Flask
`route()` without an explicit literal `methods` list is represented as GET only;
implicit HEAD/OPTIONS behavior is not modeled. Spring `RequestMapping` without an
explicit method uses an empty tuple to mean unspecified. Methods are deduplicated and
stored in the deterministic order listed above.

Only plain quoted path literals are accepted. Runtime expressions, interpolation,
escaped strings, constant propagation, and application-module execution are excluded.
Handlers are recorded only when their names are syntactically direct; anonymous
Express callbacks have no handler name. Python and Spring route spans cover the full
decorated/annotated handler declaration, while Express spans cover the registration
statement. All spans use the existing one-based inclusive line convention.

Spring class prefixes are joined only with mappings on lexically nested methods in the
same class. There is no repository-wide prefix resolution or Express router-mount
resolution. Unsupported frameworks simply produce no routes; this is not an error.

## Outbound HTTP-call extraction

Outbound calls are extracted independently from backend routes using another
in-memory Tree-sitter parse of the same whole-file text. Recognition traverses actual
call-expression and argument nodes, so call-shaped text in comments and strings is
ignored. The supported client matrix is deliberately conservative:

| Language | Client | Recognized static forms |
| --- | --- | --- |
| JavaScript/TypeScript | Fetch | `fetch(static_target)` and a direct object-literal `method` option |
| JavaScript/TypeScript | Axios | Direct `axios.get/post/put/patch/delete/head/options(static_target, ...)` calls |
| Python | Requests | Direct `requests.get/post/put/patch/delete/head/options(static_target, ...)` calls |
| Python | HTTPX | Direct `httpx.get/post/put/patch/delete/head/options(static_target, ...)` calls |

Fetch without options, or with an object literal that omits `method`, is recorded as
GET. A dynamic options value leaves the method unspecified rather than guessing.
Known methods are normalized to uppercase. Stable client labels are `fetch`, `axios`,
`requests`, and `httpx`.

Plain quoted targets are retained exactly. Simple JavaScript template placeholders
such as `` `/api/users/${id}` `` and Python f-string placeholders such as
`f"/api/users/{user_id}"` are normalized to `/api/users/{id}` and
`/api/users/{user_id}` respectively when every interpolation is a bare identifier.
Dynamic expressions, escaped literals, constant propagation, configuration lookup,
and runtime URL evaluation are not supported.

The lexical containing function name is retained when directly available. This is
not call-graph analysis. HTTP-call spans cover the narrow Tree-sitter call expression,
using the same one-based inclusive line convention as all other structural records.
Configured Axios instances, HTTPX client variables, arbitrary custom clients, and
Java HTTP clients are intentionally unsupported. They yield no metadata rather than
an error.

A backend route definition such as `@app.get("/users/{id}")` declares a server
endpoint. An outbound call such as ``axios.get(`/users/${id}`)`` invokes an HTTP
target. This stage extracts both facts independently; it does not compare, link, or
otherwise infer a relationship between them.

## Structural enrichment

`src.ingestion.enrichment.enrich_chunks(chunks, file_structure)` maps Prompt 5's
whole-file records to Prompt 4's chunks. It validates that chunks belong to the same
source file and contain positive, valid line ranges. It then uses one-based inclusive
range overlap: two records overlap when the later start line is no later than the
earlier end line. A shared boundary line therefore counts as an overlap.

Line overlap is a lightweight, deterministic bridge between whole-file syntax and
retrieval-sized chunks. It requires no text matching, compiler-grade name resolution,
or second parse. A nested method's span and its containing class span can both overlap
one chunk, so both records are retained. Likewise, a large function or class can
legitimately appear on every chunk it spans.

Imports follow exactly the same rule using their own statement spans. They are not
copied to every chunk in a file, resolved to repository paths, or used to construct a
dependency graph.

Enrichment creates new LangChain Documents in the original order. It preserves their
IDs, source text, and existing metadata, adding the plain serializable lists
`structural_definitions`, `structural_imports`, `structural_routes`, and
`structural_http_calls`. All keys are present as empty lists when a chunk has no
matches. Definitions retain name, qualified name, kind, signature, and start/end
lines. Imports retain source, imported items, alias, wildcard status, and start/end
lines. Routes retain path, uppercase methods, framework, handler, lexical owner, and
start/end lines. HTTP calls retain method, target, client, lexical caller, and
start/end lines. Duplicate identical records are removed and records are sorted by
source position with stable tie-breakers.

Routes use the same inclusive line-overlap primitive as definitions and imports. A
route can appear on multiple chunks when its handler declaration crosses a chunk
boundary. Route metadata remains separate from unchanged source `page_content`;
retrieval does not use it yet.

Outbound calls use that same overlap rule and are attached only to chunks containing
their call spans. Their metadata also remains separate from unchanged source
`page_content`.

## Why Tree-sitter is used here

LangChain remains responsible for generic Documents, text splitting, and future RAG
orchestration. Tree-sitter supplies multi-language syntax structure—named definitions,
lexical nesting, imports, and precise spans—that ordinary text splitters cannot
reliably provide. It is an analysis adapter, not a second chunking pipeline.

## Structural-analysis non-goals

The implemented structural scope is definitions, imports, selected backend routes,
selected outbound HTTP calls, and source-span mapping to chunks. It does not provide
compiler-grade semantic analysis, a complete call graph, LSP or SCIP integration, a
graph database, import-to-file resolution, or repository-wide symbol resolution.
Route-to-call matching, frontend-to-backend links, service or repository dependency
graphs, neighbor expansion, and retrieval use of structural metadata are not
implemented.

BM25, dense retrieval, embeddings, vector storage, Reciprocal Rank Fusion,
cross-encoder reranking, and LLM answer generation remain planned and are not
implemented.
