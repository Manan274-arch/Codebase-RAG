# End-to-End RAG Comparison

- Repository: `https://github.com/Manan274-arch/Title-block-and-BOM-extraction-for-deciding-raw-materials.git`
- Resolved commit: `de4a29e4d453d0695b1a2cf5d76f8285032bea70`
- Benchmark: `title-block-bom-e2e-v1`
- Questions: 18

## Overall metrics

| Metric | Dense-only baseline | Final RAG | Improvement |
| --- | ---: | ---: | ---: |
| Answer correctness | 90.3% | 88.9% | -1.4% |
| Citation precision | 58.3% | 64.0% | +5.6% |
| Citation recall | 48.0% | 64.1% | +16.1% |
| Groundedness | 86.1% | 87.5% | +1.4% |
| Supporting-code accuracy | 88.9% | 87.5% | -1.4% |
| Unanswerable accuracy | 100.0% | 100.0% | +0.0% |
| Overall success rate | 38.9% | 33.3% | -5.6% |

## Observed comparison

Final RAG improvements: Citation recall +16.1%, Citation precision +5.6%, Groundedness +1.4%.
Final RAG regressions: Overall success rate -5.6%, Answer correctness -1.4%, Supporting-code accuracy -1.4%.

## Category metrics

### simple

| Metric | Dense-only | Final RAG |
| --- | ---: | ---: |
| Answer correctness | 87.5% | 87.5% |
| Citation precision | 50.0% | 62.5% |
| Citation recall | 50.0% | 75.0% |
| Groundedness | 81.2% | 87.5% |
| Supporting-code accuracy | 93.8% | 93.8% |
| Unanswerable accuracy | N/A | N/A |
| Overall success rate | 50.0% | 50.0% |

### single-file

| Metric | Dense-only | Final RAG |
| --- | ---: | ---: |
| Answer correctness | 90.0% | 90.0% |
| Citation precision | 80.0% | 80.0% |
| Citation recall | 80.0% | 80.0% |
| Groundedness | 90.0% | 80.0% |
| Supporting-code accuracy | 80.0% | 75.0% |
| Unanswerable accuracy | N/A | N/A |
| Overall success rate | 60.0% | 60.0% |

### cross-file

| Metric | Dense-only | Final RAG |
| --- | ---: | ---: |
| Answer correctness | 95.0% | 90.0% |
| Citation precision | 56.7% | 73.3% |
| Citation recall | 36.7% | 66.7% |
| Groundedness | 85.0% | 95.0% |
| Supporting-code accuracy | 100.0% | 100.0% |
| Unanswerable accuracy | N/A | N/A |
| Overall success rate | 40.0% | 20.0% |

### architecture

| Metric | Dense-only | Final RAG |
| --- | ---: | ---: |
| Answer correctness | 75.0% | 75.0% |
| Citation precision | 83.3% | 67.5% |
| Citation recall | 40.0% | 60.0% |
| Groundedness | 75.0% | 75.0% |
| Supporting-code accuracy | 62.5% | 62.5% |
| Unanswerable accuracy | N/A | N/A |
| Overall success rate | 0.0% | 0.0% |

### unanswerable

| Metric | Dense-only | Final RAG |
| --- | ---: | ---: |
| Answer correctness | 100.0% | 100.0% |
| Citation precision | 0.0% | 0.0% |
| Citation recall | 0.0% | 0.0% |
| Groundedness | 100.0% | 100.0% |
| Supporting-code accuracy | 100.0% | 100.0% |
| Unanswerable accuracy | 100.0% | 100.0% |
| Overall success rate | 0.0% | 0.0% |

## Failure analysis

A failure means the strict overall-success rule was not met; it does not necessarily mean the answer was factually wrong.

| Question | System | Likely stage | Reason |
| --- | --- | --- | --- |
| simple-01 | production | citation mapping | The answer accurately states that DrawingPipeline is defined in src/pipeline.py and that run is the public entry point, also noting its invocation from main.py. All statements are backed by the cited snippets from src/pipeline.py and main.py, with no hallucinated or incorrect claims. |
| simple-03 | dense_only | generation | Correct but missing citation. |
| simple-04 | dense_only | generation | Mentions helper but omits sanitization details; partially grounded. |
| simple-04 | production | generation | Same partial answer; lacks full details and evidence. |
| single-01 | production | citation mapping | Correct content but no citations or evidence provided. |
| single-02 | dense_only | citation mapping | Fully covers points, but lacks explicit citations and code references. |
| single-05 | dense_only | generation | Covers atomic write but misses JSON derivation and error handling. |
| single-05 | production | generation | Shows atomic write code but omits JSON derivation and error handling. |
| cross-01 | dense_only | citation mapping | Accurate stages, early stops, export handling correct; lacks citations, so groundedness lower. |
| cross-01 | production | citation mapping | Comprehensive, cites code, matches all expected points; fully grounded and accurate. |
| cross-03 | dense_only | citation mapping | Accurately traces upload, temp dir, pipeline, export, read, session, download. |
| cross-03 | production | citation mapping | Matches all expected steps with correct code references. |
| cross-04 | dense_only | citation mapping | Fully correct, grounded, accurate. |
| cross-04 | production | citation mapping | Fully correct, grounded, accurate. |
| cross-05 | production | generation | Cites page and bbox, omits LayoutWord creation. |
| architecture-01 | dense_only | citation mapping | Comprehensive, correct, minor file list omission. |
| architecture-01 | production | citation mapping | Correct, grounded, missing some supporting file references. |
| architecture-02 | dense_only | generation | Partial coverage, missing title block, BOM, exporter evidence. |
| architecture-02 | production | generation | Missing BOMExtractor and OutputExporter details, otherwise correct. |
| unanswerable-01 | dense_only | citation mapping | Accurately states no S3 upload. |
| unanswerable-01 | production | citation mapping | Correctly notes absence of S3 logic. |
| unanswerable-02 | dense_only | citation mapping | No OCR implementation found; answer correctly states unavailability. |
| unanswerable-02 | production | citation mapping | No OCR implementation found; answer correctly states unavailability. |
## Methodology

Citation precision/recall are deterministic exact repository-relative file overlap against manually curated supporting files. A separate post-generation Groq rubric judge scores correctness, groundedness, supporting-code accuracy, and hallucination from 0-4; its complete bounded inputs and raw outputs are saved, while results retain every full displayed snippet. The two answer systems use identical default 1,024-token generation settings; the post-generation structured judge uses a 2,048-token output allowance. Overall answerable success requires >=75% correctness, >=75% citation precision, >=50% citation recall, >=75% groundedness, >=75% supporting-code accuracy, and no material hallucination. Unanswerable success additionally requires a correct refusal/qualification while retaining those same citation, groundedness, and supporting-code thresholds.

Detailed answers, citations, snippets, retrieval/context identifiers, latencies, judge inputs, raw judge outputs, and failure classifications are retained in `results.json`.
