# RAG Evaluation Report

## Summary

- **total_questions**: 61.0000
- **answerable_questions**: 51.0000
- **unanswerable_questions**: 10.0000
- **avg_source_match**: 1.0000
- **avg_keyword_recall**: 1.0000
- **avg_context_keyword_recall**: 1.0000
- **avg_answer_groundedness**: 1.0000
- **avg_hallucination_risk**: 0.0000
- **avg_ragas_context_precision**: 1.0000
- **avg_ragas_context_recall**: 1.0000
- **avg_ragas_faithfulness**: 1.0000
- **avg_ragas_answer_relevancy**: 1.0000
- **avg_unanswerable_safe**: 1.0000
- **overall_pass_rate**: 1.0000

## OpenCompass-Style Evaluation Framing

This project follows an OpenCompass-style separation between dataset, inferencer, evaluator, and report artifacts:
- **Dataset**: bilingual English/Chinese policy questions with expected sources and keywords.
- **Inferencer**: the RAG pipeline retrieves top-k chunks and generates extractive answers.
- **Retriever**: hybrid BM25 + local dense embeddings rank bilingual context chunks.
- **Evaluator**: rule-based and RAGAS-style metrics score source match, keyword recall, context recall, groundedness, hallucination risk, faithfulness, answer relevancy, and unanswerable safety.
- **Reporter**: CSV and Markdown outputs summarize aggregate metrics and failed-case diagnostics.
Unlike a leaderboard benchmark, this framework is optimized for RAG QA regression: it checks whether answers stay grounded in the project knowledge base and whether unsupported questions are refused.

## Failed Cases

No failed cases were found.


## Chinese Failed-Case Analysis

No Chinese failed cases were found.
