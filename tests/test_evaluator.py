import pandas as pd
from src.evaluator import keyword_recall, source_match, answerability_check, add_pass_fail_flags, identify_failed_cases


def test_keyword_recall_full_match():
    answer = "Customers can return items within 30 days of delivery."
    keywords = "30 days;return;delivery"
    assert keyword_recall(answer, keywords) == 1.0


def test_keyword_recall_partial_match():
    answer = "Customers can return items within 30 days."
    keywords = "30 days;return;delivery"
    assert keyword_recall(answer, keywords) == 2 / 3


def test_source_match_success():
    assert source_match(["return_policy.md", "shipping_policy.md"], "return_policy.md") == 1


def test_source_match_failure():
    assert source_match(["shipping_policy.md"], "return_policy.md") == 0


def test_unanswerable_safe_answer():
    answer = "I could not find enough information in the provided documents."
    assert answerability_check(answer, "unanswerable") == 1


def test_add_pass_fail_flags():
    df = pd.DataFrame([
        {"source_match": 1, "keyword_recall": 0.8, "unanswerable_safe": None},
        {"source_match": 0, "keyword_recall": 0.8, "unanswerable_safe": None},
    ])
    result = add_pass_fail_flags(df)
    assert result["overall_pass"].iloc[0] == 1
    assert result["overall_pass"].iloc[1] == 0


def test_identify_failed_cases():
    df = pd.DataFrame([
        {"question": "Q1", "source_match": 1, "keyword_recall": 0.8, "unanswerable_safe": None},
        {"question": "Q2", "source_match": 0, "keyword_recall": 0.8, "unanswerable_safe": None},
    ])
    flagged = add_pass_fail_flags(df)
    failed = identify_failed_cases(flagged)
    assert len(failed) == 1
    assert failed["question"].iloc[0] == "Q2"
