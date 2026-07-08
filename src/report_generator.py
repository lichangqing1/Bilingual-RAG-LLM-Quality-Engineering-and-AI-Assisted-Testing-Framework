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
    lines.append("\n## Failed Cases\n")
    if failed_cases_df.empty:
        lines.append("No failed cases were found.\n")
    else:
        for i, row in failed_cases_df.iterrows():
            lines.append(f"### Failed Case {i + 1}\n")
            lines.append(f"- **Question**: {row.get('question', '')}")
            lines.append(f"- **Failure Type**: {row.get('failure_type', '')}")
            lines.append(f"- **Expected Source**: {row.get('expected_source', '')}")
            lines.append(f"- **Retrieved Sources**: {row.get('retrieved_sources', '')}")
            lines.append(f"- **Keyword Recall**: {row.get('keyword_recall', '')}")
            lines.append(f"- **Missing Keywords**: {row.get('missing_keywords', '')}")
            lines.append(f"- **Answer**: {row.get('answer', '')}\n")
    report_text = "\n".join(lines)
    output_file.write_text(report_text, encoding="utf-8")
    return str(output_file)
