# Multi-Language Codebase Q&A RAG Assistant

This project will answer natural-language questions about software repositories by
retrieving relevant source-code context and using it to generate referenced answers.

Repository source discovery, LangChain Document loading, language-aware code
chunking, structural extraction for definitions, imports, routes, and outbound HTTP
calls, and conservative route-to-call relationship matching are implemented. The
shared retrieval representation contains a compact structural summary followed by the
exact original code. BM25 and local code-aware dense retrieval consume that same text.
Rank-only Reciprocal Rank Fusion combines their candidate lists into hybrid results.
The hybrid candidates are reranked locally with a cross-encoder. Answer generation
remains planned. An offline regression benchmark measures every retrieval stage with
Hit Rate, Recall, MRR, and nDCG at multiple cutoffs.

```text
Enriched Shared Code Corpus
      /       \
   BM25       Dense
      \       /
        RRF
         |
 Hybrid Candidates
         |
 Cross-Encoder
         |
 Final Ranked Results
```

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
use the local model cache. Retrieval is fully local and embeds the shared structural
summary plus original code. Exact raw code remains in `metadata["raw_content"]`.

Run the BM25, dense, and hybrid RRF comparison on the unchanged benchmark:

```shell
python -m src.retrieval.evaluate_hybrid
```

Run RRF and the real local cross-encoder comparison:

```shell
python -m src.retrieval.evaluate_reranker
```

| Retriever | Hit Rate@1 | Hit Rate@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.8000 | 0.9333 | 0.8444 | 0.8667 |
| Dense | 0.8000 | 1.0000 | 0.8889 | 0.9175 |
| RRF | 0.8667 | 0.9333 | 0.9000 | 0.9087 |

Compared with the frozen raw-code results, enriched BM25 gains most on relationship
queries, dense is unchanged overall, and untuned RRF now leads Hit Rate@1 and MRR@3.
Dense still leads Hit Rate@3 and nDCG@3 before reranking.

The default `cross-encoder/ms-marco-MiniLM-L6-v2` reranker produces:

| Retriever | Hit Rate@1 | Hit Rate@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.8000 | 1.0000 | 0.8889 | 0.9175 |
| RRF | 0.8667 | 0.9333 | 0.9000 | 0.9087 |
| RRF + Cross-Encoder | 0.9333 | 1.0000 | 0.9667 | 0.9754 |

The reranker scores only RRF candidates using enriched `page_content`; it does not
scan the corpus. It runs locally and preserves exact source in `raw_content`.
Relationship expansion and answer generation remain unimplemented.
