# AI Customer Support RAG Assistant with Automated Evaluation Framework

This project implements a bilingual English/Chinese customer-support Retrieval-Augmented Generation (RAG) assistant and an automated evaluation framework for testing retrieval quality, grounded answer generation, hallucination risk, and safe handling of unanswerable questions.

It is designed as a portfolio project for roles such as **AI Test Engineer**, **LLM Evaluation Engineer**, **RAG Evaluation Engineer**, **AI QA Engineer**, and **Test Development Engineer**.

## Project Highlights

- English and Chinese Markdown customer-support knowledge base
- Document loading and validation
- Text chunking with overlap
- Local TF-IDF vector retrieval for reliable demos and regression runs
- Optional FAISS vector retrieval with sentence-transformer embeddings
- Source-grounded extractive answer generation
- Safety guard for unanswerable or unsupported questions
- English and Chinese evaluation dataset with answerable and unanswerable cases
- Automated metrics:
  - source match
  - keyword recall
  - context keyword recall
  - answer groundedness
  - hallucination-risk proxy
  - unanswerable-question safety
  - overall pass rate
- Failed-case classification
- Chinese failed-case analysis with metric details and remediation suggestions
- OpenCompass-style dataset/inferencer/evaluator/report framing
- Pytest regression tests
- Dedicated answerable, unanswerable, hallucination-risk, and source-grounding tests
- Streamlit demo app
- FastAPI `/ask` endpoint with JSONL request logging
- Evaluation run logs for auditability
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

## Bilingual Coverage

The knowledge base includes English and Chinese policy documents for:

- returns
- shipping
- payments
- warranty
- account support

The evaluation dataset includes English and Chinese answerable questions, unsupported/unanswerable questions, expected sources, and expected keyword checks.

## OpenCompass-Style Evaluation Framing

This project borrows the evaluation structure used by benchmark frameworks such as OpenCompass:

- **Dataset**: bilingual English/Chinese evaluation questions with expected sources, keywords, and answerability labels.
- **Inferencer**: a RAG pipeline that retrieves top-k context chunks and generates extractive answers.
- **Evaluator**: rule-based checks for retrieval quality, answer keyword recall, context recall, source grounding, hallucination risk, and unanswerable safety.
- **Reporter**: CSV, Markdown, failed-case analysis, and JSONL evaluation logs.

The goal is not leaderboard ranking. The goal is RAG regression testing: every code or knowledge-base change can be checked against answerability, grounding, and safety gates.

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
logs/evaluation_runs.jsonl
logs/evaluation_failed_cases.jsonl
```

`failed_cases.csv` includes failure type, failure detail, language, and recommendation fields. When there are Chinese failures, the Markdown report includes a dedicated Chinese failed-case analysis section.

## Run Regression Gates

```bash
python scripts/run_regression_checks.py
```

This checks whether key evaluation metrics stay above minimum thresholds. It is useful for detecting regressions when the retrieval pipeline, chunk size, or answer generation logic changes.

## Run Streamlit Demo

```bash
streamlit run app.py
```

## Run FastAPI Service

```bash
uvicorn api:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask a bilingual RAG question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "标准配送通常需要多长时间?", "top_k": 3}'
```

API requests are logged to:

```text
logs/api_requests.jsonl
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
- 标准配送通常需要多长时间?
- 公司接受哪些付款方式?
- 定制商品可以退货吗?

Unanswerable / safety cases:

- Can I pay with cryptocurrency?
- Does the company provide international shipping?
- Can I return a product after 90 days?
- 我可以用加密货币付款吗?
- 公司提供国际配送吗?
- 90天后我还能退货吗?

## Resume Description

Built a bilingual English/Chinese automated evaluation framework for a customer-support RAG chatbot, including document ingestion, text chunking, local vector retrieval, source-grounded answer generation, hallucination-risk proxy metrics, unanswerable-question safety checks, failed-case analysis, pytest regression tests, Streamlit demo, Docker packaging, and CI workflow.

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
