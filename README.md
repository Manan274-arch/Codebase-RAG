# Multi-Language Codebase Q&A RAG Assistant

This project will answer natural-language questions about software repositories by
retrieving relevant source-code context and using it to generate referenced answers.

Repository source discovery, LangChain Document loading, and language-aware code
chunking are implemented. Indexing, retrieval, and answer generation are not yet
implemented.

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
