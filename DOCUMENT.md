# Technical Architecture

## Project objective

Build a multi-language Codebase Q&A RAG Assistant capable of answering
natural-language questions about a software repository using retrieved source-code
context.

## Planned high-level pipeline

```text
repository
-> source-file discovery
-> LangChain Documents
-> language-aware code splitting
-> embeddings
-> vector store
-> retrieval
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

Repository source-file discovery is implemented in `src/ingestion/repository.py`.
Its responsibility is to recursively identify candidate source files while pruning
version-control metadata, virtual environments, dependency/vendor directories,
generated build output, caches, coverage output, and editor metadata. Symbolic-link
directories are not traversed.

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
representation. Discovery relies only on file extensions: files are not read,
inspected for binary content, parsed, or converted into LangChain Documents.

Chunking, embeddings, a vector database, retrieval, and LLM integration remain
unimplemented.
