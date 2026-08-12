# Bilingual RAG Evaluation and Security Testing Framework

A bilingual English/Chinese RAG evaluation framework for testing retrieval quality, answer grounding, hallucination risk, unanswerable-question handling, and security behavior.

This project is built as a portfolio-ready example for AI Testing, LLM Evaluation, RAG Evaluation, AI QA, and AI Application Engineering roles.

## Overview

The framework evaluates a customer-support RAG system over English and Chinese policy documents. It checks whether the system can:

- retrieve the correct supporting policy context;
- answer only when the knowledge base supports the answer;
- cite the retrieved source;
- safely refuse unsupported questions;
- detect security-sensitive prompts such as prompt injection, jailbreaks, and system-prompt leakage.

The benchmark is deterministic and intentionally compact. Its purpose is regression testing and evaluation framework design, not leaderboard comparison.

## Key Features

| Area | What is included |
|---|---|
| Bilingual RAG | English and Chinese documents, questions, and expected answers |
| Evaluation cases | Answerable QA, unanswerable QA, hallucination checks, source-grounding checks |
| Security testing | Prompt injection, jailbreak, system prompt leakage, sensitive information disclosure, retrieval poisoning, unsafe instruction refusal |
| Retrieval modes | Lexical baseline, semantic retrieval, and hybrid retrieval |
| Metrics | Faithfulness, context recall, context precision, answer relevancy, citation accuracy, security pass rate |
| Interfaces | Streamlit demo and FastAPI service |
| Engineering workflow | Pytest tests, evaluation scripts, regression gates, GitHub Actions CI |

## Architecture

```text
Documents -> Chunks -> Retriever -> RAG Pipeline -> Evaluator -> Reports
                                      |
                                      +-> Streamlit Demo
                                      +-> FastAPI Service
                                      +-> JSONL Logs
```

Main source layout:

```text
src/
├── document_loader.py
├── text_splitter.py
├── retrieval.py
├── rag_pipeline.py
├── evaluator.py
├── report_generator.py
└── retrievers/
    ├── lexical_retriever.py
    ├── semantic_retriever.py
    └── hybrid_retriever.py
```

Detailed design notes are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Evaluation Results

Current deterministic benchmark result:

| Metric | Value |
|---|---:|
| `overall_pass_rate` | `1.0000` |
| `security_pass_rate` | `1.0000` |
| `avg_faithfulness` | `1.0000` |
| `avg_context_recall` | `1.0000` |
| `avg_context_precision` | `1.0000` |

The current prepared benchmark reaches 100% on the small bilingual customer-support evaluation set. This is best understood as a regression gate for the prepared cases, not as a public benchmark score.

![Evaluation report](docs/screenshots/evaluation_report.png)

Security evaluation:

![Security evaluation](docs/screenshots/security_evaluation.png)

Generated result files:

```text
results/evaluation_report.md
results/summary_report.csv
results/evaluation_results.csv
results/security_summary.csv
results/security_evaluation_results.csv
results/failed_cases.csv
```

`failed_cases.csv` is empty in the current run because all prepared benchmark cases passed.

For metric definitions, formulas, pass examples, fail examples, and limitations, see [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md). For safety-specific evaluation details, see [docs/SECURITY_EVALUATION.md](docs/SECURITY_EVALUATION.md).

## Retrieval Modes

Default mode: `hybrid`.

| Retrieval mode | Context recall | Context precision | Answer groundedness | Latency | Notes |
|---|---:|---:|---:|---|---|
| Keyword only | 1.00 | 1.00 | 1.00 | fast | Good for exact policy terms |
| Semantic FAISS | 1.00 | 1.00 | 1.00 | medium | Better for paraphrased questions |
| Hybrid | 1.00 | 1.00 | 1.00 | medium | Best overall balance |

Implementation summary:

| Mode | Implementation | Use case |
|---|---|---|
| `lexical` / `keyword` | TF-IDF-style keyword retrieval | Fast baseline and CI-friendly fallback |
| `semantic` | Sentence-Transformers + FAISS, or local semantic backend | Main semantic retrieval path |
| `hybrid` | Lexical + semantic score fusion | Default balanced mode |

The default install is lightweight. FAISS and Sentence-Transformers are optional:

```bash
pip install -r requirements-vector.txt
python scripts/run_evaluation.py --retrieval-mode semantic --semantic-backend faiss
```

## How to Run

Create a Python 3.10+ environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the evaluation workflow:

```bash
python scripts/run_evaluation.py
python scripts/run_security_evaluation.py
python scripts/run_regression_checks.py
```

If your terminal has multiple Python versions, use the venv Python directly:

```bash
.venv/bin/python scripts/run_evaluation.py
.venv/bin/python scripts/run_security_evaluation.py
.venv/bin/python scripts/run_regression_checks.py
```

## API Demo

Start the FastAPI service:

```bash
uvicorn api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

![FastAPI Swagger demo](docs/screenshots/fastapi_swagger.png)

Example `/ask` request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long does standard shipping take?", "retrieval_mode": "hybrid", "semantic_backend": "local"}'
```

Main endpoints:

```text
GET  /health
POST /ask
POST /evaluate
POST /feedback
GET  /metrics
GET  /logs/summary
```

## Streamlit Demo

Start the Streamlit app:

```bash
streamlit run app.py
```

![Streamlit demo](docs/screenshots/streamlit_demo.png)

The Streamlit app defaults to the lightweight local backend. If you choose the FAISS backend, install optional vector dependencies first:

```bash
pip install -r requirements-vector.txt
```

## Test and CI

Run local tests:

```bash
pytest -q
```

![Pytest passed](docs/screenshots/pytest_passed.png)

Run regression gates:

```bash
python scripts/run_evaluation.py
python scripts/run_security_evaluation.py
python scripts/run_regression_checks.py
```

![Regression checks](docs/screenshots/regression_checks.png)

GitHub Actions runs tests, regenerates evaluation results, runs security evaluation, and then checks regression gates. This prevents the CI gate from passing only because of old committed result files.

## Project Structure

```text
rag-evaluation-framework/
├── .github/workflows/ci.yml
├── configs/rag_eval_config.yaml
├── data/
│   ├── documents/
│   └── evaluation/
│       ├── evaluation_questions.csv
│       ├── rag_eval_en.csv
│       ├── rag_eval_zh.csv
│       └── security_questions.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION_METRICS.md
│   ├── PROJECT_PORTFOLIO_CN.md
│   ├── SECURITY_EVALUATION.md
│   └── screenshots/
├── results/
├── scripts/
├── src/
│   └── retrievers/
├── tests/
├── api.py
├── app.py
├── Dockerfile
├── requirements.txt
└── requirements-vector.txt
```

## Job Application Value

This project shows practical experience with:

- bilingual RAG test-data design;
- retrieval comparison across lexical, semantic, and hybrid modes;
- deterministic RAGAS-style evaluation metrics;
- hallucination-risk and citation-accuracy checks;
- safety testing for common LLM attack patterns;
- production-style API endpoints and request logging;
- CI-based regression gates.

Chinese portfolio summary: [docs/PROJECT_PORTFOLIO_CN.md](docs/PROJECT_PORTFOLIO_CN.md).
