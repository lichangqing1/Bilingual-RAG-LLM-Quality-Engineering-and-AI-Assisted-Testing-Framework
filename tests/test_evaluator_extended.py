import pandas as pd

from src.evaluator import (
    answer_groundedness,
    add_pass_fail_flags,
    detect_language,
    context_keyword_recall,
    identify_failed_cases,
    hallucination_risk,
    evaluate_single_case,
)


def test_context_keyword_recall():
    context = "Standard shipping usually takes 3 to 5 business days."
    assert context_keyword_recall(context, "standard shipping;3 to 5 business days") == 1.0


def test_answer_groundedness_detects_supported_answer():
    context = "Refunds are issued to the original payment method."
    answer = "Refunds are issued to the original payment method. Source: payment_policy.md."
    assert answer_groundedness(answer, context) >= 0.8
    assert hallucination_risk(answer, context) <= 0.2


def test_evaluate_single_case_includes_new_metrics():
    result = {
        "answer": "Standard shipping usually takes 3 to 5 business days. Source: shipping_policy.md.",
        "sources": ["shipping_policy.md"],
        "retrieved_context": "Standard shipping usually takes 3 to 5 business days.",
    }
    expected = pd.Series({
        "question": "How long does standard shipping take?",
        "expected_answer": "Standard shipping usually takes 3 to 5 business days.",
        "expected_source": "shipping_policy.md",
        "expected_keywords": "standard shipping;3 to 5 business days",
        "question_type": "normal",
    })
    row = evaluate_single_case(result, expected)
    assert row["source_match"] == 1
    assert row["context_keyword_recall"] == 1.0
    assert row["answer_groundedness"] >= 0.8


def test_detect_language_supports_chinese_and_english():
    assert detect_language("标准配送通常需要多长时间?") == "zh"
    assert detect_language("How long does standard shipping take?") == "en"


def test_failed_case_analysis_includes_detail_and_recommendation():
    df = pd.DataFrame([{
        "question": "公司提供国际配送吗?",
        "language": "zh",
        "question_type": "unanswerable",
        "answer": "公司提供国际配送。",
        "expected_source": "none",
        "retrieved_sources": "shipping_policy_zh.md",
        "source_match": None,
        "keyword_recall": 0.0,
        "context_keyword_recall": None,
        "answer_groundedness": None,
        "hallucination_risk": None,
        "unanswerable_safe": 0,
    }])
    failed = identify_failed_cases(add_pass_fail_flags(df))

    assert len(failed) == 1
    assert "failure_detail" in failed.columns
    assert "recommendation" in failed.columns
    assert "unanswerable question was not safely refused" in failed.iloc[0]["failure_detail"]
