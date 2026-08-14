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
-> shared chunk corpus
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
not discarded. No custom AST, Tree-sitter, Shell, or SQL parsing is used.

Every chunk preserves all parent metadata and adds:

- `chunk_index`: Zero-based within its source Document and restarted for each file.
- `start_index`: Character offset of the chunk in the complete source text, supplied
  by LangChain's public text-splitter API.

Empty source Documents produce no chunks. Non-empty chunks contain source text only;
file paths and language labels remain metadata. Splitter whitespace stripping is
disabled so source whitespace is preserved. Parent and within-parent ordering are
preserved, making the output a deterministic shared corpus for future sparse and dense
retrieval.

BM25, dense retrieval, embeddings, vector storage, Reciprocal Rank Fusion,
cross-encoder reranking, and LLM answer generation remain planned and are not
implemented.
