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
```

The remaining planned pipeline is:

```text
-> language-aware code splitting
-> shared chunk corpus
   /              \
BM25          dense retrieval
   \              /
 -> Reciprocal Rank Fusion (RRF)
 -> cross-encoder reranking
-> LLM answer generation
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

Repository source-file discovery and LangChain Document creation are implemented in
`src/ingestion/repository.py` and `src/ingestion/loader.py`. Discovery recursively
identifies candidate source files while pruning version-control metadata, virtual
environments, dependency/vendor directories, generated build output, caches, coverage
output, and editor metadata. Symbolic-link directories are not traversed.

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

No chunking exists yet. The shared chunk corpus, BM25 and dense retrieval, embeddings,
vector storage, Reciprocal Rank Fusion, cross-encoder reranking, and LLM answer
generation are planned but not implemented.
