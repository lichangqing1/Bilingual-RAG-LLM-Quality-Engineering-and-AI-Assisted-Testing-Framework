"""Run the full RAG evaluation workflow from the command line."""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_markdown_documents, validate_documents
from src.text_splitter import create_chunks
from src.tfidf_vector_store import TfidfVectorStore
from src.rag_pipeline import SimpleRAGPipeline
from src.evaluator import evaluate_dataset, save_evaluation_outputs, identify_failed_cases, summarize_results
from src.logging_utils import append_jsonl
from src.report_generator import generate_markdown_report


def main() -> None:
    docs = load_markdown_documents(str(PROJECT_ROOT / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    vector_store = TfidfVectorStore()
    vector_store.build_index(chunks)
    rag = SimpleRAGPipeline(vector_store, top_k=3)

    evaluation_df = pd.read_csv(PROJECT_ROOT / "data" / "evaluation" / "evaluation_questions.csv")
    evaluation_results = evaluate_dataset(rag, evaluation_df)
    outputs = save_evaluation_outputs(evaluation_results, output_dir=str(PROJECT_ROOT / "results"))

    failed_cases = identify_failed_cases(evaluation_results)
    summary = summarize_results(evaluation_results)
    report_path = generate_markdown_report(summary, failed_cases, str(PROJECT_ROOT / "results" / "evaluation_report.md"))
    summary_record = summary.iloc[0].to_dict()
    append_jsonl(
        PROJECT_ROOT / "logs" / "evaluation_runs.jsonl",
        {
            "event": "evaluation_run_completed",
            "total_cases": int(summary_record.get("total_questions", 0)),
            "answerable_cases": int(summary_record.get("answerable_questions", 0)),
            "unanswerable_cases": int(summary_record.get("unanswerable_questions", 0)),
            "overall_pass_rate": float(summary_record.get("overall_pass_rate", 0)),
            "avg_unanswerable_safe": float(summary_record.get("avg_unanswerable_safe", 0)),
            "failed_cases": int(len(failed_cases)),
            "report_path": report_path,
        },
    )
    for _, row in failed_cases.iterrows():
        append_jsonl(
            PROJECT_ROOT / "logs" / "evaluation_failed_cases.jsonl",
            {
                "event": "evaluation_failed_case",
                "question": row.get("question", ""),
                "language": row.get("language", ""),
                "question_type": row.get("question_type", ""),
                "failure_type": row.get("failure_type", ""),
                "failure_detail": row.get("failure_detail", ""),
                "recommendation": row.get("recommendation", ""),
            },
        )

    print("Evaluation completed.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"- markdown_report: {report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
