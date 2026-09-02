"""Run the RAG robustness challenge evaluation suite."""
from __future__ import annotations

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

from ai_testing.executor import build_generated_case_rag
from src.evaluation.evaluator import add_pass_fail_flags, evaluate_dataset, identify_failed_cases, summarize_results
from src.logging_utils import append_jsonl


CHALLENGE_DATASET = PROJECT_ROOT / "data" / "evaluation" / "challenge_questions.csv"


def print_header(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}")
    print(title)
    print(line)


def print_subheader(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_key_values(values: Dict[str, object]) -> None:
    width = max((len(key) for key in values), default=0)
    for key, value in values.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        print(f"{key:<{width}} : {value}")


def load_challenge_questions(dataset_path: Path = CHALLENGE_DATASET) -> pd.DataFrame:
    """Load challenge cases for robustness/generalization checks."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Challenge dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    required = ["question", "expected_answer", "expected_source", "expected_keywords", "question_type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Challenge dataset is missing columns: {missing}")
    return df


def run_challenge_evaluation(
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
    top_k: int = 3,
    output_dir: Path = PROJECT_ROOT / "results",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run challenge cases and save row-level, failed-case, and summary outputs."""
    challenge_df = load_challenge_questions()
    rag = build_generated_case_rag(
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
        top_k=top_k,
    )
    results = add_pass_fail_flags(evaluate_dataset(rag, challenge_df))
    failed_cases = identify_failed_cases(results)
    summary = summarize_results(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "challenge_evaluation_results.csv", index=False)
    failed_cases.to_csv(output_dir / "challenge_failed_cases.csv", index=False)
    summary.to_csv(output_dir / "challenge_summary.csv", index=False)

    append_jsonl(
        PROJECT_ROOT / "logs" / "challenge_evaluation_runs.jsonl",
        {
            "event": "challenge_evaluation_completed",
            "retrieval_mode": retrieval_mode,
            "semantic_backend": semantic_backend,
            "top_k": top_k,
            "total_cases": int(summary.iloc[0]["total_questions"]),
            "overall_pass_rate": float(summary.iloc[0]["overall_pass_rate"]),
            "failed_cases": int(len(failed_cases)),
        },
    )
    return results, failed_cases, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG challenge evaluation.")
    parser.add_argument("--retrieval-mode", choices=["lexical", "keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--semantic-backend", choices=["local", "faiss"], default="local")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    _, failed_cases, summary = run_challenge_evaluation(
        retrieval_mode=args.retrieval_mode,
        semantic_backend=args.semantic_backend,
        top_k=args.top_k,
    )
    summary_record = summary.iloc[0].to_dict()

    print_header("Challenge Evaluation Completed")
    print_subheader("Run Configuration")
    print_key_values(
        {
            "Dataset": CHALLENGE_DATASET,
            "Retrieval mode": args.retrieval_mode,
            "Semantic backend": args.semantic_backend,
            "Top k": args.top_k,
        }
    )

    print_subheader("Output Files")
    print(f"- challenge_evaluation_results: {PROJECT_ROOT / 'results' / 'challenge_evaluation_results.csv'}")
    print(f"- challenge_failed_cases: {PROJECT_ROOT / 'results' / 'challenge_failed_cases.csv'}")
    print(f"- challenge_summary: {PROJECT_ROOT / 'results' / 'challenge_summary.csv'}")

    print_subheader("Summary Metrics")
    print_key_values(summary_record)
    print(f"- failed_cases: {len(failed_cases)}")


if __name__ == "__main__":
    main()
