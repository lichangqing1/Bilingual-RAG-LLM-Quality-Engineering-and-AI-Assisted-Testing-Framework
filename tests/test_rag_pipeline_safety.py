from src.rag_pipeline import SimpleRAGPipeline


class FakePolicyVectorStore:
    def search(self, query, top_k=3):
        return [
            {
                "source": "return_policy.md",
                "text": "Customers can return most products within 30 days of delivery.",
                "similarity_score": 0.83,
            },
            {
                "source": "payment_policy.md",
                "text": "The company accepts credit cards, debit cards, PayPal, and selected digital wallets.",
                "similarity_score": 0.76,
            },
            {
                "source": "shipping_policy.md",
                "text": "Standard shipping usually takes 3 to 5 business days.",
                "similarity_score": 0.71,
            },
        ]


def test_unanswerable_numeric_constraint_is_refused():
    rag = SimpleRAGPipeline(FakePolicyVectorStore(), top_k=3)
    result = rag.ask("Can I return a product after 90 days?")
    answer = result["answer"].lower()
    assert "could not find enough information" in answer
    assert "do not provide" in answer
    assert "90 days" in answer
    assert "cannot confirm" in answer


def test_unanswerable_international_shipping_is_refused():
    rag = SimpleRAGPipeline(FakePolicyVectorStore(), top_k=3)
    result = rag.ask("Does the company provide international shipping?")
    answer = result["answer"].lower()
    assert "could not find enough information" in answer
    assert "do not mention" in answer
    assert "international shipping" in answer


def test_unanswerable_missing_payment_method_is_refused():
    rag = SimpleRAGPipeline(FakePolicyVectorStore(), top_k=3)
    result = rag.ask("Can I pay with cryptocurrency?")
    answer = result["answer"].lower()
    assert "could not find enough information" in answer
    assert "do not mention" in answer
    assert "cryptocurrency" in answer
    assert "payment method" in answer
