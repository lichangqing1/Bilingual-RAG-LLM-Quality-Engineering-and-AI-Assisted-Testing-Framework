# Architecture

```text
configs/rag_eval_config.yaml
        |
        v
Evaluation Dataset Files
        |-- evaluation_questions.csv
        |-- rag_eval_en.csv
        |-- rag_eval_zh.csv
        |-- security_questions.csv
        |
        v
English + Chinese Markdown Documents
        |
        v
Document Loader -> Validator
        |
        v
Text Splitter -> Chunks
        |
        v
Retriever Factory
        |-- lexical: BM25 / TF-IDF only
        |-- semantic: Sentence-Transformers + FAISS
        |-- hybrid: lexical + semantic + score fusion
        |-- local semantic backend for offline regression runs
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
        |-- context_precision
        |-- context_recall
        |-- faithfulness
        |-- answer_relevancy
        |-- citation_accuracy
        |-- faithfulness_failure_hallucination_risk
        |-- unanswerable_safe
        |-- security_pass_rate
        |-- source_match (legacy)
        |-- keyword_recall
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
- prompt injection, jailbreak, system prompt leakage, sensitive-data requests, retrieval poisoning, and unsafe bypass instructions;
- low question-evidence keyword coverage.

This design is intentionally conservative because the project focuses on AI testing and RAG evaluation rather than maximum answer generation.

## Bilingual evaluation design

The evaluation set includes English and Chinese rows for:

- answerable policy questions;
- unanswerable or unsupported questions;
- prompt-injection and policy-override security questions;
- source-match checks;
- keyword-recall checks;
- retrieved-context keyword checks;
- source-grounding and hallucination-risk checks.

The main production semantic retriever uses Sentence-Transformers embeddings with FAISS vector search. Those vector dependencies live in `requirements-vector.txt` so the default `requirements.txt` remains lightweight for CI and simple local setup. A lightweight lexical retriever is retained as a deterministic baseline and CI-friendly fallback. The default hybrid retriever combines lexical scoring with a local semantic backend for repeatable offline tests, while `semantic_backend=faiss` enables production-style semantic retrieval and hybrid fusion after installing the optional vector dependencies.

## OpenCompass-style evaluation mapping

The project is organized like a compact RAG benchmark:

- **Config**: `configs/rag_eval_config.yaml` declares dataset files, task types, metrics, outputs, and quality gates.
- **Dataset**: `scripts/run_evaluation.py` loads the benchmark-style shards declared in config: `rag_eval_en.csv`, `rag_eval_zh.csv`, and `security_questions.csv`. `evaluation_questions.csv` remains as a legacy combined fallback.
- **Inferencer**: `SimpleRAGPipeline` retrieves top-k chunks with the configured retrieval mode: lexical, semantic, or hybrid.
- **Evaluator**: `src/evaluator.py` computes context precision, context recall, faithfulness, answer relevancy, citation accuracy, faithfulness failure / hallucination risk, safety, and backward-compatible legacy metric aliases.
- **Reporter**: `src/report_generator.py` creates Markdown reports, failed-case diagnostics, Chinese failed-case analysis, and summary CSV files.
- **Logs**: API requests and evaluation runs are written as JSONL for auditability.

## API and logging

`api.py` exposes:

- `GET /health`
- `POST /ask`
- `POST /evaluate`
- `POST /feedback`
- `GET /metrics`
- `GET /logs/summary`

Each `/ask` request logs question, top-k setting, answer, sources, and latency to `logs/api_requests.jsonl`.
Each `/evaluate` request logs row-level evaluation metrics to `logs/api_evaluations.jsonl`.
Each `/feedback` request logs user feedback to `logs/feedback.jsonl`.

Running `scripts/run_evaluation.py` logs each evaluation run to `logs/evaluation_runs.jsonl` and failed cases to `logs/evaluation_failed_cases.jsonl`.

`GET /metrics` returns evaluation summary metrics plus per-log line counts.
`GET /logs/summary` returns a compact summary for each JSONL log file, including existence, line count, last timestamp, and last event.
