"""Minimal regression gate for RAG evaluation results."""
import os
from pathlib import Path
import sys

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def print_gate_row(metric: str, value: float, minimum: float, passed: bool) -> None:
    """Print one aligned regression gate result."""
    status = "PASS" if passed else "FAIL"
    print(f"{status:<4}  {metric:<32} value={value:.4f}  minimum={minimum:.4f}")


def main() -> None:
    summary_path = PROJECT_ROOT / "results" / "summary_report.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run scripts/run_evaluation.py before regression checks.")

    summary = pd.read_csv(summary_path).iloc[0]
    gates = {
        "overall_pass_rate": 0.75,
        "avg_source_match": 0.75,
        "avg_keyword_recall": 0.60,
        "avg_unanswerable_safe": 0.80,
        "avg_ragas_context_precision": 0.75,
        "avg_ragas_context_recall": 0.75,
        "avg_ragas_faithfulness": 0.80,
        "avg_ragas_answer_relevancy": 0.60,
    }

    failed = []
    rows = []
    for metric, minimum in gates.items():
        value = float(summary.get(metric, 0))
        passed = value >= minimum
        rows.append((metric, value, minimum, passed))
        if value < minimum:
            failed.append(f"{metric}={value:.3f} < {minimum:.3f}")

    print_header("Regression Gate Results")
    print_subheader("Input")
    print(f"- summary_report: {summary_path}")

    print_subheader("Gate Checks")
    for metric, value, minimum, passed in rows:
        print_gate_row(metric, value, minimum, passed)

    if failed:
        print_subheader("Result")
        print("FAILED")
        raise AssertionError("Regression gate failed: " + "; ".join(failed))

    print_subheader("Result")
    print("PASSED - all regression gates passed.")


if __name__ == "__main__":
    main()
