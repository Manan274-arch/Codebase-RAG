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

After Prompt 1, only project scaffolding and development tooling exist. There is
currently no repository ingestion, chunking, embedding model, vector database,
retriever, or LLM integration.
