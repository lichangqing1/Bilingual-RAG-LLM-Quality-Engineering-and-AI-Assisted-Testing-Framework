import pandas as pd

from src.evaluator import (
    answer_groundedness,
    context_keyword_recall,
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
