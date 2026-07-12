# Architecture

```text
English + Chinese Markdown Documents
        |
        v
Document Loader -> Validator
        |
        v
Text Splitter -> Chunks
        |
        v
Hybrid Retriever (default)
        |-- BM25 lexical retrieval
        |-- local dense embeddings from TF-IDF + SVD
        |
        | optional
        v
FAISS Vector Store + Sentence Transformer Embeddings
        |
        v
Retriever -> Top-K Context Chunks
        |
        v
Extractive Answer Generator with Safety Guard
        |
        +--------------------+
        |                    |
        v                    v
Streamlit Demo        FastAPI Production Endpoints
                             |
                             v
                    JSONL Request Logs

Evaluation Framework
        |-- source_match
        |-- keyword_recall
        |-- context_keyword_recall
        |-- answer_groundedness
        |-- hallucination_risk
        |-- ragas_context_precision
        |-- ragas_context_recall
        |-- ragas_faithfulness
        |-- ragas_answer_relevancy
        |-- unanswerable_safe
        |
        v
CSV Reports + Failed Case Analysis + Markdown Report + JSONL Evaluation Logs
```

## Safety guard logic

The pipeline refuses to answer when retrieved evidence is insufficient. The current rule-based guard checks:

- retrieval similarity below threshold;
- missing numeric constraints such as `90 days`;
- high-signal missing terms such as `cryptocurrency` or `international`;
- Chinese unsupported terms such as `加密货币` and `国际配送`;
- low question-evidence keyword coverage.

This design is intentionally conservative because the project focuses on AI testing and RAG evaluation rather than maximum answer generation.

## Bilingual evaluation design

The evaluation set includes English and Chinese rows for:

- answerable policy questions;
- unanswerable or unsupported questions;
- source-match checks;
- keyword-recall checks;
- retrieved-context keyword checks;
- source-grounding and hallucination-risk checks.

The default hybrid retriever combines BM25 lexical scoring with local dense embeddings built from TF-IDF + SVD. It keeps the demo lightweight while supporting English and Chinese retrieval without external embedding downloads.

## OpenCompass-style evaluation mapping

The project is organized like a compact RAG benchmark:

- **Dataset**: `data/evaluation/evaluation_questions.csv` stores bilingual prompts, expected answers, expected sources, expected keywords, and answerability labels.
- **Inferencer**: `SimpleRAGPipeline` retrieves top-k chunks with hybrid BM25 + embedding retrieval and produces extractive, source-cited answers.
- **Evaluator**: `src/evaluator.py` computes retrieval, answer, grounding, hallucination-risk, safety, and RAGAS-style proxy metrics.
- **Reporter**: `src/report_generator.py` creates Markdown reports, failed-case diagnostics, Chinese failed-case analysis, and summary CSV files.
- **Logs**: API requests and evaluation runs are written as JSONL for auditability.

## API and logging

`api.py` exposes:

- `GET /health`
- `POST /ask`
- `POST /evaluate`
- `POST /feedback`
- `GET /metrics`

Each `/ask` request logs question, top-k setting, answer, sources, and latency to `logs/api_requests.jsonl`.
Each `/evaluate` request logs row-level evaluation metrics to `logs/api_evaluations.jsonl`.
Each `/feedback` request logs user feedback to `logs/feedback.jsonl`.

Running `scripts/run_evaluation.py` logs each evaluation run to `logs/evaluation_runs.jsonl` and failed cases to `logs/evaluation_failed_cases.jsonl`.
