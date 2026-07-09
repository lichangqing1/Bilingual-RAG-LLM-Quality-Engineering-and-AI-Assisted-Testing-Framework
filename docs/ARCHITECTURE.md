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
TF-IDF Vector Store (default)
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
        v
Evaluation Framework
        |-- source_match
        |-- keyword_recall
        |-- context_keyword_recall
        |-- answer_groundedness
        |-- hallucination_risk
        |-- unanswerable_safe
        |
        v
CSV Reports + Failed Case Analysis + Markdown Report
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

The default TF-IDF retriever uses character n-grams, which keeps the demo lightweight while supporting English and Chinese retrieval without external embedding downloads.
