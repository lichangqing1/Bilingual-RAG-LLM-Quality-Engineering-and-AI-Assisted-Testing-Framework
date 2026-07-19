# Bilingual RAG Evaluation and Security Testing Framework

`rag-evaluation-framework` is a bilingual English/Chinese customer-support RAG project with automated evaluation for retrieval quality, faithfulness, citation accuracy, unanswerable-question handling, and security behavior.

It is designed as a portfolio project for AI Test Engineer, LLM Evaluation Engineer, RAG Evaluation Engineer, AI QA Engineer, and AI Application Engineer roles.

## Overview

The project builds a source-grounded RAG assistant over English and Chinese policy documents, then evaluates the assistant with deterministic regression tests. The goal is not leaderboard comparison; the goal is repeatable RAG quality testing whenever retrieval, chunking, prompts, or knowledge-base files change.

## Key Features

- Bilingual English/Chinese customer-support knowledge base
- Lexical, semantic, and hybrid retrieval modes
- Main semantic path: Sentence-Transformers + FAISS
- Lightweight default install for CI and Colab
- Source-cited extractive answers
- Safe refusal for unsupported questions
- RAGAS-style deterministic metrics
- Security evaluation with `security_pass_rate`
- Streamlit demo and FastAPI service
- Pytest and GitHub Actions CI

## Architecture

```text
Documents -> Chunks -> Retriever -> RAG Answer -> Evaluator -> Reports
```

The retriever package is organized as:

```text
src/retrievers/
├── base.py
├── lexical_retriever.py
├── semantic_retriever.py
└── hybrid_retriever.py
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full technical design.

## Evaluation Results

Current deterministic benchmark:

```text
overall_pass_rate: 1.00
security_pass_rate: 1.00
```

The prepared benchmark currently reaches a 100% pass rate on the small bilingual customer-support evaluation set. This is intended as a regression gate for retrieval, grounding, citation, unanswerable handling, and safety behavior, not as a public leaderboard score.

Generated result files:

```text
results/evaluation_report.md
results/summary_report.csv
results/security_summary.csv
results/evaluation_results.csv
results/security_evaluation_results.csv
results/failed_cases.csv
```

`failed_cases.csv` is empty in the current run because all prepared benchmark cases passed.

Metric details live in [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md).

## Retrieval Modes

| Mode | Implementation | Notes |
|---|---|---|
| `lexical` | BM25 / TF-IDF | Fast baseline and CI-friendly fallback |
| `semantic` | Sentence-Transformers + FAISS, or local semantic backend | Better for paraphrased questions |
| `hybrid` | Lexical + semantic score fusion | Default mode and best overall balance |

The default backend is local and lightweight. To use FAISS:

```bash
pip install -r requirements-vector.txt
python scripts/run_evaluation.py --retrieval-mode semantic --semantic-backend faiss
```

## How To Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Run full evaluation:

```bash
python scripts/run_evaluation.py
```

Run security evaluation:

```bash
python scripts/run_security_evaluation.py
```

Run regression gates:

```bash
python scripts/run_regression_checks.py
```

For Colab, use [docs/COLAB_EXECUTION.md](docs/COLAB_EXECUTION.md).

## API Demo

Run FastAPI locally:

```bash
uvicorn api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long does standard shipping take?", "retrieval_mode": "hybrid", "semantic_backend": "local"}'
```

Available endpoints:

```text
GET  /health
POST /ask
POST /evaluate
POST /feedback
GET  /metrics
GET  /logs/summary
```

## Streamlit Demo

```bash
streamlit run app.py
```

The Streamlit UI defaults to the local backend. Selecting FAISS in the UI requires:

```bash
pip install -r requirements-vector.txt
```

## Docker

The default Dockerfile starts the Streamlit demo:

```bash
docker build -t rag-evaluation-framework .
docker run --rm -p 8501:8501 rag-evaluation-framework
```

Open:

```text
http://localhost:8501
```

To run FastAPI locally, use:

```bash
uvicorn api:app --reload --port 8000
```

Or run FastAPI from the same Docker image:

```bash
docker run --rm -p 8000:8000 rag-evaluation-framework \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

## Test And CI

GitHub Actions installs `requirements.txt`, then runs:

```bash
pytest -q
python scripts/run_evaluation.py
python scripts/run_security_evaluation.py
python scripts/run_regression_checks.py
```

This ensures the regression gate uses freshly generated evaluation results.

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
│   ├── COLAB_EXECUTION.md
│   ├── EVALUATION_METRICS.md
│   ├── PROJECT_PORTFOLIO_CN.md
│   └── SECURITY_EVALUATION.md
├── results/
│   ├── evaluation_report.md
│   ├── summary_report.csv
│   └── security_summary.csv
├── scripts/
│   ├── run_evaluation.py
│   ├── run_security_evaluation.py
│   └── run_regression_checks.py
├── src/
│   ├── retrievers/
│   ├── rag_pipeline.py
│   ├── evaluator.py
│   └── report_generator.py
├── tests/
├── api.py
├── app.py
├── Dockerfile
├── requirements.txt
└── requirements-vector.txt
```

## Job Application Value

This project demonstrates practical RAG engineering and AI testing skills: bilingual data preparation, retrieval comparison, deterministic evaluation design, hallucination-risk checks, source grounding, API logging, safety regression testing, CI gates, and deployable demos.
