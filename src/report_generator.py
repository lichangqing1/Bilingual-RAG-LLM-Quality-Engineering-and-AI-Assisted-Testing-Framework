from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_markdown_report(summary_df: pd.DataFrame, failed_cases_df: pd.DataFrame, output_path: str = "results/evaluation_report.md") -> str:
    """Generate a simple Markdown evaluation report."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    lines = ["# RAG Evaluation Report\n", "## Summary\n"]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- **{key}**: {value:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.append("\n## OpenCompass-Style Evaluation Framing\n")
    lines.append(
        "This project follows an OpenCompass-style separation between dataset, "
        "inferencer, evaluator, and report artifacts:"
    )
    lines.append("- **Dataset**: bilingual English/Chinese policy questions with expected sources and keywords.")
    lines.append("- **Inferencer**: the RAG pipeline retrieves top-k chunks and generates extractive answers.")
    lines.append("- **Retriever**: configurable lexical baseline, semantic FAISS path, or hybrid score-fusion retrieval ranks bilingual context chunks.")
    lines.append("- **Evaluator**: rule-based and RAGAS-style metrics score context precision, context recall, faithfulness, answer relevancy, citation accuracy, faithfulness failure / hallucination risk, and unanswerable safety.")
    lines.append("- **Reporter**: CSV and Markdown outputs summarize aggregate metrics and failed-case diagnostics.")
    lines.append(
        "Unlike a leaderboard benchmark, this framework is optimized for RAG QA regression: "
        "it checks whether answers stay grounded in the project knowledge base and whether unsupported questions are refused."
    )
    lines.append("\n## Failed Cases\n")
    if failed_cases_df.empty:
        lines.append("No failed cases were found.\n")
    else:
        for i, row in failed_cases_df.iterrows():
            lines.append(f"### Failed Case {i + 1}\n")
            lines.append(f"- **Question**: {row.get('question', '')}")
            lines.append(f"- **Language**: {row.get('language', '')}")
            lines.append(f"- **Failure Type**: {row.get('failure_type', '')}")
            lines.append(f"- **Failure Detail**: {row.get('failure_detail', '')}")
            lines.append(f"- **Recommendation**: {row.get('recommendation', '')}")
            lines.append(f"- **Expected Source**: {row.get('expected_source', '')}")
            lines.append(f"- **Retrieved Sources**: {row.get('retrieved_sources', '')}")
            lines.append(f"- **Context Precision**: {row.get('context_precision', '')}")
            lines.append(f"- **Context Recall**: {row.get('context_recall', '')}")
            lines.append(f"- **Faithfulness**: {row.get('faithfulness', '')}")
            lines.append(f"- **Answer Relevancy**: {row.get('answer_relevancy', '')}")
            lines.append(f"- **Citation Accuracy**: {row.get('citation_accuracy', '')}")
            lines.append(f"- **Keyword Recall**: {row.get('keyword_recall', '')}")
            lines.append(f"- **Missing Keywords**: {row.get('missing_keywords', '')}")
            lines.append(f"- **Answer**: {row.get('answer', '')}\n")
    chinese_failed_cases = failed_cases_df[failed_cases_df.get("language", pd.Series(dtype=str)) == "zh"] if not failed_cases_df.empty else pd.DataFrame()
    lines.append("\n## Chinese Failed-Case Analysis\n")
    if chinese_failed_cases.empty:
        lines.append("No Chinese failed cases were found.\n")
    else:
        lines.append(f"- Chinese failed cases: {len(chinese_failed_cases)}")
        for _, row in chinese_failed_cases.iterrows():
            lines.append(f"- `{row.get('question', '')}`: {row.get('failure_detail', '')}")
    report_text = "\n".join(lines)
    output_file.write_text(report_text, encoding="utf-8")
    return str(output_file)
