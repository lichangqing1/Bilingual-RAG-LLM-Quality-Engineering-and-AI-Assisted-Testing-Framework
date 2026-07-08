# AI Customer Support RAG Assistant with Automated Evaluation Framework

This project implements a customer-support Retrieval-Augmented Generation (RAG) assistant and an automated evaluation framework for testing retrieval quality, grounded answer generation, hallucination risk, and safe handling of unanswerable questions.

It is designed as a portfolio project for roles such as **AI Test Engineer**, **LLM Evaluation Engineer**, **RAG Evaluation Engineer**, **AI QA Engineer**, and **Test Development Engineer**.

## Project Highlights

- Markdown customer-support knowledge base
- Document loading and validation
- Text chunking with overlap
- Local TF-IDF vector retrieval for reliable demos and regression runs
- Optional FAISS vector retrieval with sentence-transformer embeddings
- Source-grounded extractive answer generation
- Safety guard for unanswerable or unsupported questions
- Evaluation dataset with answerable and unanswerable cases
- Automated metrics:
  - source match
  - keyword recall
  - context keyword recall
  - answer groundedness
  - hallucination-risk proxy
  - unanswerable-question safety
  - overall pass rate
- Failed-case classification
- Pytest regression tests
- Streamlit demo app
- Dockerfile and GitHub Actions CI
- China-facing portfolio notes in `docs/PROJECT_PORTFOLIO_CN.md`

## Project Structure

```text
.
├── app.py
├── data/
│   ├── documents/
│   └── evaluation/evaluation_questions.csv
├── docs/
│   ├── ARCHITECTURE.md
│   └── PROJECT_PORTFOLIO_CN.md
├── scripts/
│   ├── run_evaluation.py
│   └── run_regression_checks.py
├── src/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── tfidf_vector_store.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── evaluator.py
│   └── report_generator.py
├── tests/
├── results/
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run Unit Tests

```bash
pytest -q
```

The default retrieval path uses scikit-learn TF-IDF so tests and demos run without FAISS, PyTorch, or Hugging Face downloads.

## Run Full Evaluation

```bash
python scripts/run_evaluation.py
```

The script writes:

```text
results/evaluation_results.csv
results/failed_cases.csv
results/summary_report.csv
results/evaluation_report.md
```

## Run Regression Gates

```bash
python scripts/run_regression_checks.py
```

This checks whether key evaluation metrics stay above minimum thresholds. It is useful for detecting regressions when the retrieval pipeline, chunk size, or answer generation logic changes.

## Run Streamlit Demo

```bash
streamlit run app.py
```

## Run with Docker

```bash
docker build -t rag-evaluation-framework .
docker run -p 8501:8501 rag-evaluation-framework
```

Then open:

```text
http://localhost:8501
```

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| `source_match` | Whether the expected document appears in retrieved sources |
| `keyword_recall` | Whether the generated answer contains expected keywords |
| `context_keyword_recall` | Whether retrieved context contains expected keywords |
| `answer_groundedness` | Proxy score for whether answer claims are supported by retrieved context |
| `hallucination_risk` | `1 - answer_groundedness` |
| `unanswerable_safe` | Whether unsupported questions are safely refused |
| `overall_pass_rate` | Combined pass rate across applicable checks |

## Example Questions

Answerable:

- How long does standard shipping usually take?
- What payment methods does the company accept?
- Can customized products be returned?

Unanswerable / safety cases:

- Can I pay with cryptocurrency?
- Does the company provide international shipping?
- Can I return a product after 90 days?

## Resume Description

Built an automated evaluation framework for a customer-support RAG chatbot, including document ingestion, text chunking, local vector retrieval, source-grounded answer generation, hallucination-risk proxy metrics, unanswerable-question safety checks, failed-case analysis, pytest regression tests, Streamlit demo, Docker packaging, and CI workflow.

## China-Facing Positioning

Recommended Chinese project title:

> RAG智能问答系统评测与自动化测试框架

Recommended target roles:

- 大模型测试工程师
- AI测试开发工程师
- 大模型评测工程师
- RAG评测工程师
- 算法测试工程师
- 测试开发工程师

See `docs/PROJECT_PORTFOLIO_CN.md` for a Chinese resume-style explanation.
