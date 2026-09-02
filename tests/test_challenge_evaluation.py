from pathlib import Path

import pandas as pd

from scripts.run_challenge_evaluation import load_challenge_questions, run_challenge_evaluation


def test_challenge_dataset_has_expected_schema_and_case_mix():
    df = load_challenge_questions()

    assert list(df.columns) == [
        "question",
        "expected_answer",
        "expected_source",
        "expected_keywords",
        "question_type",
    ]
    assert len(df) >= 20
    assert {"normal", "unanswerable"}.issubset(set(df["question_type"]))
    assert any("45" in question for question in df["question"])
    assert any("国际配送" in question for question in df["question"])
    assert any("warranty" in question.lower() for question in df["question"])


def test_challenge_evaluation_writes_outputs(tmp_path):
    _, _, summary = run_challenge_evaluation(output_dir=tmp_path)

    assert (tmp_path / "challenge_evaluation_results.csv").exists()
    assert (tmp_path / "challenge_failed_cases.csv").exists()
    assert (tmp_path / "challenge_summary.csv").exists()
    saved_summary = pd.read_csv(tmp_path / "challenge_summary.csv")
    assert saved_summary.iloc[0]["total_questions"] == summary.iloc[0]["total_questions"]
