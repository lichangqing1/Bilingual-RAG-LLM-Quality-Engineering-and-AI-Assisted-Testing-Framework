# Architecture

```text
Markdown Documents
        |
        v
Document Loader -> Validator
        |
        v
Text Splitter -> Chunks
        |
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
- low question-evidence keyword coverage.

This design is intentionally conservative because the project focuses on AI testing and RAG evaluation rather than maximum answer generation.
