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


def print_gate_row(metric: str, value: float, threshold: float, passed: bool) -> None:
    """Print one aligned regression gate result."""
    status = "PASS" if passed else "FAIL"
    label = "maximum" if metric.startswith("max_") else "minimum"
    print(f"{status:<4}  {metric:<52} value={value:.4f}  {label}={threshold:.4f}")


def load_quality_gates(config_path: Path = PROJECT_ROOT / "configs" / "rag_eval_config.yaml") -> dict[str, float]:
    """Load quality gates from the project config without adding a YAML parser."""
    defaults = {
        "overall_pass_rate": 0.90,
        "avg_source_match": 0.75,
        "avg_keyword_recall": 0.60,
        "avg_unanswerable_safe": 0.80,
        "avg_context_precision": 0.75,
        "avg_context_recall": 0.75,
        "avg_faithfulness": 0.80,
        "avg_answer_relevancy": 0.60,
        "max_avg_faithfulness_failure_hallucination_risk": 0.20,
    }
    if not config_path.exists():
        return defaults

    gates = defaults.copy()
    in_quality_gates = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        if indent == 0:
            in_quality_gates = line == "quality_gates:"
            continue
        if in_quality_gates and indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            try:
                gates[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return gates


def main() -> None:
    summary_path = PROJECT_ROOT / "results" / "summary_report.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run scripts/run_evaluation.py before regression checks.")

    summary = pd.read_csv(summary_path).iloc[0]
    gates = load_quality_gates()

    failed = []
    rows = []
    for metric, minimum in gates.items():
        if metric.startswith("max_"):
            summary_metric = metric.removeprefix("max_")
            value = float(summary.get(summary_metric, 0))
            passed = value <= minimum
        else:
            summary_metric = metric
            value = float(summary.get(summary_metric, 0))
            passed = value >= minimum
        rows.append((metric, value, minimum, passed))
        if not passed:
            comparator = ">" if metric.startswith("max_") else "<"
            failed.append(f"{summary_metric}={value:.3f} {comparator} {minimum:.3f}")

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
