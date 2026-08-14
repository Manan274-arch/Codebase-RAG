# Multi-Language Codebase Q&A RAG Assistant

This project will answer natural-language questions about software repositories by
retrieving relevant source-code context and using it to generate referenced answers.

Repository source discovery, LangChain Document loading, language-aware code
chunking, and metadata-only enrichment for definitions, imports, and selected backend
route and outbound HTTP-client patterns are implemented. Relationship matching,
including conservative route-to-call chunk links, is implemented. A raw-source-only
BM25 lexical baseline and a local code-aware dense retriever are also available.
Hybrid retrieval and answer generation remain planned. An offline regression
benchmark measures both retrievers with Hit Rate, Recall, MRR, and nDCG at multiple
cutoffs.

## Development setup

Python 3.11 is required.

```shell
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the development checks:

```shell
pytest
ruff check .
mypy src
```

Reproduce the committed BM25 retrieval baseline:

```shell
python -m src.retrieval.evaluate_bm25
```

Run the real local dense model and compare it with BM25 on the same frozen corpus:

```shell
python -m src.retrieval.evaluate_dense
```

The first dense run downloads `jinaai/jina-embeddings-v2-base-code`; subsequent runs
use the local model cache. Retrieval is fully local and embeds raw chunk source only.
