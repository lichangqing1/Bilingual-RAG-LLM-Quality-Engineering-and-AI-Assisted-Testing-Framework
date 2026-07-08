"""Minimal regression gate for RAG evaluation results."""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    }

    failed = []
    for metric, minimum in gates.items():
        value = float(summary.get(metric, 0))
        if value < minimum:
            failed.append(f"{metric}={value:.3f} < {minimum:.3f}")

    if failed:
        raise AssertionError("Regression gate failed: " + "; ".join(failed))

    print("All regression gates passed.")


if __name__ == "__main__":
    main()
