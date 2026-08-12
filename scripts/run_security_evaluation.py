"""Run the dedicated RAG safety and security evaluation suite."""
import argparse
import os
from pathlib import Path
import sys
from typing import Dict

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_markdown_documents, validate_documents
from src.evaluator import add_pass_fail_flags, evaluate_dataset, is_security_case
from src.logging_utils import append_jsonl
from src.rag_pipeline import SimpleRAGPipeline
from src.retrieval import build_vector_store
from src.text_splitter import create_chunks


SECURITY_DATASET = PROJECT_ROOT / "data" / "evaluation" / "security_questions.csv"


def print_header(title: str) -> None:
    """Print a readable CLI section header."""
    line = "=" * 72
    print(f"\n{line}")
    print(title)
    print(line)


def print_subheader(title: str) -> None:
    """Print a readable CLI subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def print_key_values(values: Dict[str, object]) -> None:
    """Print aligned key/value pairs for command-line reports."""
    width = max((len(key) for key in values), default=0)
    for key, value in values.items():
        print(f"{key:<{width}} : {value}")


def format_metric(value: object) -> str:
    """Format summary metrics consistently."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_security_questions(dataset_path: Path = SECURITY_DATASET) -> pd.DataFrame:
    """Load the dedicated security evaluation CSV."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Security dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    required = ["question", "expected_answer", "expected_source", "expected_keywords", "question_type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Security dataset is missing columns: {missing}")
    invalid_types = sorted({str(value) for value in df["question_type"] if not is_security_case(value)})
    if invalid_types:
        raise ValueError(f"Unsupported security question_type values: {invalid_types}")
    return df


def build_security_rag(
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
    top_k: int = 3,
) -> SimpleRAGPipeline:
    """Build the default RAG pipeline used by the security evaluation."""
    docs = load_markdown_documents(str(PROJECT_ROOT / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = build_vector_store(
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
    )
    vector_store.build_index(chunks)
    return SimpleRAGPipeline(vector_store, top_k=top_k)


def summarize_security_results(evaluation_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize security evaluation results with one primary pass-rate metric."""
    df = evaluation_results.copy()
    required = ["question_type", "unanswerable_safe"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Security results are missing columns: {missing}")
    if "overall_pass" not in df.columns and "source_match" in df.columns:
        df = add_pass_fail_flags(df)

    category_summary = (
        df.groupby("question_type")["unanswerable_safe"]
        .agg(["count", "mean"])
        .rename(columns={"count": "case_count", "mean": "pass_rate"})
        .reset_index()
    )
    security_pass_rate = df["unanswerable_safe"].mean(skipna=True)
    summary: Dict[str, object] = {
        "total_security_cases": int(len(df)),
        "failed_security_cases": int((df["unanswerable_safe"] != 1).sum()),
        "security_pass_rate": float(security_pass_rate) if not pd.isna(security_pass_rate) else 0.0,
    }
    for _, row in category_summary.iterrows():
        summary[f"{row['question_type']}_case_count"] = int(row["case_count"])
        summary[f"{row['question_type']}_pass_rate"] = float(row["pass_rate"])
    return pd.DataFrame([summary])


def run_security_evaluation(
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
    top_k: int = 3,
    output_dir: Path = PROJECT_ROOT / "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the security suite and save result artifacts."""
    security_df = load_security_questions()
    rag = build_security_rag(
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
        top_k=top_k,
    )
    results = add_pass_fail_flags(evaluate_dataset(rag, security_df))
    summary = summarize_security_results(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "security_evaluation_results.csv", index=False)
    summary.to_csv(output_dir / "security_summary.csv", index=False)

    append_jsonl(
        PROJECT_ROOT / "logs" / "security_evaluation_runs.jsonl",
        {
            "event": "security_evaluation_completed",
            "retrieval_mode": retrieval_mode,
            "semantic_backend": semantic_backend,
            "top_k": top_k,
            "total_security_cases": int(summary.iloc[0]["total_security_cases"]),
            "failed_security_cases": int(summary.iloc[0]["failed_security_cases"]),
            "security_pass_rate": float(summary.iloc[0]["security_pass_rate"]),
        },
    )
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dedicated RAG security evaluation.")
    parser.add_argument("--retrieval-mode", choices=["lexical", "keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--semantic-backend", choices=["local", "faiss"], default="local")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    _, summary = run_security_evaluation(
        retrieval_mode=args.retrieval_mode,
        semantic_backend=args.semantic_backend,
        top_k=args.top_k,
    )
    print_header("Security Evaluation Completed")
    print_subheader("Run Configuration")
    print_key_values(
        {
            "Dataset": SECURITY_DATASET,
            "Retrieval mode": args.retrieval_mode,
            "Semantic backend": args.semantic_backend,
            "Top k": args.top_k,
        }
    )

    print_subheader("Output Files")
    print(f"- security_evaluation_results: {PROJECT_ROOT / 'results' / 'security_evaluation_results.csv'}")
    print(f"- security_summary: {PROJECT_ROOT / 'results' / 'security_summary.csv'}")

    print_subheader("Summary Metrics")
    summary_record = summary.iloc[0].to_dict()
    print_key_values({key: format_metric(value) for key, value in summary_record.items()})


if __name__ == "__main__":
    main()
