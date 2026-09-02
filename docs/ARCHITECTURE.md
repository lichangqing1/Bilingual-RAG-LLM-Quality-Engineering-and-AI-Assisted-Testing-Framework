# Architecture

This project has two connected layers:

- **RAG evaluation**: tests the behavior of the bilingual RAG assistant.
- **AI-assisted testing**: generates structured test assets and quality summaries for the evaluation workflow.

It intentionally avoids large agent frameworks. The project is organized around reproducible test infrastructure: schemas, deterministic evaluation, generated assets, logs, and CI gates.

## System View

```text
configs/rag_eval_config.yaml
        |
        v
Evaluation Datasets
        |-- rag_eval_en.csv
        |-- rag_eval_zh.csv
        |-- security_questions.csv
        |-- challenge_questions.csv
        |-- evaluation_questions.csv (legacy fallback)
        |
        v
English + Chinese Markdown Documents
        |
        v
Document Loader -> Validator -> Text Splitter
        |
        v
Retriever Factory
        |-- lexical: keyword baseline
        |-- semantic: local backend or Sentence-Transformers + FAISS
        |-- hybrid: lexical + semantic score fusion
        |
        v
SimpleRAGPipeline
        |-- top-k context retrieval
        |-- extractive answer generation
        |-- source citation
        |-- conservative safety guard
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
Streamlit Demo        FastAPI Service       Evaluation Scripts
                            |                   |
                            v                   v
                    JSONL Runtime Logs     CSV/Markdown Reports
```

## AI-Assisted Testing View

```text
Requirement / Feature Description
        |
        v
ai_testing.requirement_parser
        |
        v
Pydantic RequirementSpec
        |
        v
ai_testing.scenario_generator
        |
        v
Structured TestScenario objects
        |
        v
ai_testing.test_data_generator
        |
        v
GeneratedTestCase objects
        |
        +-> ai_generated_eval_cases.csv
        +-> generated_pytest_cases.py
        |
        v
RAG Evaluation Results
        |
        +-> failure_analyzer.py
        +-> rca_analyzer.py
        +-> quality_summary.py
```

The important engineering choice is that generated content is validated as structured data before it becomes a reusable test asset.

`RuleBasedGenerator` is the default domain-specific baseline for offline and CI execution. `LLMTestGenerator` uses the optional `OpenAICompatibleJSONClient` for LLM-backed requirement interpretation when `--generator llm` is explicitly requested.

LLM-assisted RCA is a separate optional enrichment layer. Deterministic metrics and quality gates still decide CI pass/fail; the LLM only receives failed-case evidence and returns Pydantic-validated diagnostic guidance.

## Retrieval Design

The retriever factory lives in `src/retrieval.py`.

| Mode | Implementation | Role |
|---|---|---|
| `lexical` / `keyword` | Lightweight TF-IDF-style retriever | Deterministic baseline and CI fallback |
| `semantic` | Local semantic backend or FAISS + Sentence-Transformers | Main semantic retrieval path |
| `hybrid` | Lexical + semantic score fusion | Default mode |

FAISS and Sentence-Transformers are optional and live in `requirements-vector.txt`. This keeps the default `requirements.txt` fast and reliable for CI.

## Evaluation Design

The framework follows an OpenCompass-style separation:

| Layer | Project implementation |
|---|---|
| Config | `configs/rag_eval_config.yaml` |
| Dataset | CSV files under `data/evaluation/` |
| Inferencer | `src/rag_pipeline.py` |
| Evaluator | `src/evaluator.py` |
| Reporter | `src/report_generator.py` |
| Regression gate | `scripts/run_regression_checks.py` |
| Challenge suite | `scripts/run_challenge_evaluation.py` |
| Optional RCA | `ai_testing/rca_analyzer.py` |

The evaluator computes deterministic proxies for:

- context precision;
- context recall;
- faithfulness;
- answer relevancy;
- citation accuracy;
- faithfulness failure / hallucination risk;
- unanswerable safe rate;
- security pass rate.

Regression thresholds are loaded from `quality_gates` in `configs/rag_eval_config.yaml`, so local checks and CI use the same source of truth. The regression script uses `yaml.safe_load()` when PyYAML is installed and retains a minimal fallback parser for lightweight environments.

## Evaluation Suites

| Suite | Dataset | Runner | Purpose |
|---|---|---|---|
| Regression | `rag_eval_en.csv`, `rag_eval_zh.csv`, `security_questions.csv` | `scripts/run_evaluation.py` | Known expected RAG behavior |
| Security | `security_questions.csv` | `scripts/run_security_evaluation.py` | Adversarial and unsafe-request refusal |
| Challenge | `challenge_questions.csv` | `scripts/run_challenge_evaluation.py` | Robustness and generalization across paraphrases, unsupported constraints, and bilingual queries |
| Generated assets | `test_assets/generated_cases/ai_generated_cases.json` | `tests/test_generated_assets.py` | Requirement-driven generated cases |

Metric details are in [EVALUATION_METRICS.md](EVALUATION_METRICS.md).

## Safety Guard

The RAG pipeline refuses when evidence is weak or the request is unsafe. The guard checks:

- low retrieval score;
- unsupported numeric constraints such as `90 days`;
- high-signal absent terms such as `cryptocurrency` and `international`;
- Chinese unsupported terms such as `加密货币` and `国际配送`;
- prompt injection, jailbreak, system-prompt leakage, sensitive data, retrieval poisoning, and unsafe bypass requests;
- weak overlap between the question and retrieved evidence.

Security details are in [SECURITY_EVALUATION.md](SECURITY_EVALUATION.md).

## API and Logging

`api.py` exposes:

```text
GET  /health
POST /ask
POST /evaluate
POST /feedback
GET  /metrics
GET  /logs/summary
POST /testing/generate
```

Runtime logs are JSONL files under `logs/`:

| Log | Source |
|---|---|
| `api_requests.jsonl` | `/ask` |
| `api_evaluations.jsonl` | `/evaluate` |
| `feedback.jsonl` | `/feedback` |
| `ai_testing.jsonl` | `/testing/generate` |
| `evaluation_runs.jsonl` | `scripts/run_evaluation.py` |
| `evaluation_failed_cases.jsonl` | failed-case logging |

`GET /metrics` returns summary metrics plus log counts. `GET /logs/summary` returns file-level log health, line counts, and last events.
