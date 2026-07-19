from pathlib import Path

import pandas as pd

from scripts.run_security_evaluation import (
    load_security_questions,
    run_security_evaluation,
    summarize_security_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_SECURITY_CATEGORIES = {
    "prompt_injection",
    "jailbreak",
    "system_prompt_leakage",
    "sensitive_information_disclosure",
    "retrieval_poisoning",
    "unsafe_instruction_refusal",
}


def test_security_dataset_covers_required_categories():
    security_df = load_security_questions(PROJECT_ROOT / "data" / "evaluation" / "security_questions.csv")

    assert set(security_df["question_type"]) == EXPECTED_SECURITY_CATEGORIES
    assert len(security_df) >= len(EXPECTED_SECURITY_CATEGORIES)
    assert security_df["question"].str.len().min() > 0
    assert security_df["expected_answer"].str.len().min() > 0


def test_security_summary_metric_is_security_pass_rate():
    results = pd.DataFrame(
        [
            {"question_type": "prompt_injection", "unanswerable_safe": 1},
            {"question_type": "jailbreak", "unanswerable_safe": 1},
            {"question_type": "retrieval_poisoning", "unanswerable_safe": 0},
        ]
    )

    summary = summarize_security_results(results).iloc[0]

    assert summary["total_security_cases"] == 3
    assert summary["failed_security_cases"] == 1
    assert summary["security_pass_rate"] == 2 / 3


def test_security_evaluation_runner_passes_default_suite():
    results, summary = run_security_evaluation()

    assert len(results) == 12
    assert set(results["question_type"]) == EXPECTED_SECURITY_CATEGORIES
    assert summary.iloc[0]["security_pass_rate"] == 1.0
    assert (PROJECT_ROOT / "results" / "security_evaluation_results.csv").exists()
    assert (PROJECT_ROOT / "results" / "security_summary.csv").exists()
