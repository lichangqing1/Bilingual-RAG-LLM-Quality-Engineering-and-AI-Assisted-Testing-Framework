import pandas as pd

from src.document_loader import load_markdown_documents, validate_documents
from src.evaluator import (
    answer_groundedness,
    evaluate_single_case,
    hallucination_risk,
)
from src.rag_pipeline import SimpleRAGPipeline
from src.retrievers.hybrid_retriever import HybridRetriever
from src.text_splitter import create_chunks


def build_bilingual_rag():
    docs = load_markdown_documents("data/documents")
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = HybridRetriever()
    vector_store.build_index(chunks)
    return SimpleRAGPipeline(vector_store, top_k=3)


def test_chinese_answerable_question_returns_grounded_answer():
    rag = build_bilingual_rag()

    result = rag.ask("标准配送通常需要多长时间?")

    assert "3到5个工作日" in result["answer"]
    assert "shipping_policy_zh.md" in result["sources"]
    assert answer_groundedness(
        result["answer"],
        result["retrieved_context"],
        question_type="normal",
    ) >= 0.8


def test_chinese_unanswerable_international_shipping_is_refused():
    rag = build_bilingual_rag()

    result = rag.ask("公司提供国际配送吗?")
    answer = result["answer"]

    assert "没有提到国际配送" in answer
    assert "international shipping" not in answer.lower()


def test_chinese_unanswerable_crypto_payment_is_refused():
    rag = build_bilingual_rag()

    result = rag.ask("我可以用加密货币付款吗?")
    answer = result["answer"]

    assert "没有提到加密货币" in answer
    assert "付款方式" in answer


def test_source_grounding_flags_supported_and_unsupported_answers():
    context = "标准配送通常需要3到5个工作日。"
    supported = "标准配送通常需要3到5个工作日。 Source: shipping_policy_zh.md."
    unsupported = "标准配送通常需要当天送达。 Source: shipping_policy_zh.md."

    assert answer_groundedness(supported, context) >= 0.8
    assert hallucination_risk(supported, context) <= 0.2
    assert answer_groundedness(unsupported, context) < 0.8
    assert hallucination_risk(unsupported, context) > 0.2


def test_evaluator_scores_chinese_answerable_case():
    result = {
        "answer": "退款会退回到原始付款方式。 Source: payment_policy_zh.md.",
        "sources": ["payment_policy_zh.md"],
        "retrieved_context": "退款会退回到原始付款方式。",
    }
    expected = pd.Series({
        "question": "退款会退到哪里?",
        "expected_answer": "退款会退回到原始付款方式。",
        "expected_source": "payment_policy_zh.md",
        "expected_keywords": "退款;原始付款方式",
        "question_type": "normal",
    })

    row = evaluate_single_case(result, expected)

    assert row["source_match"] == 1
    assert row["keyword_recall"] == 1.0
    assert row["context_keyword_recall"] == 1.0
    assert row["answer_groundedness"] >= 0.8
