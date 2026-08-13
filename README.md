# Multi-Language Codebase Q&A RAG Assistant

This project will answer natural-language questions about software repositories by
retrieving relevant source-code context and using it to generate referenced answers.

Repository source-file discovery is implemented. Source loading, document creation,
chunking, indexing, retrieval, and answer generation are not implemented yet.

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
