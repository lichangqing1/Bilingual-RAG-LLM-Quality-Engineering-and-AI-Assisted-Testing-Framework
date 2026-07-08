class FakeVectorStore:
    def search(self, query, top_k=3):
        return [
            {
                "source": "shipping_policy.md",
                "text": "Standard shipping usually takes 3 to 5 business days.",
                "similarity_score": 0.8,
            }
        ]


def test_rag_pipeline_returns_answer():
    from src.rag_pipeline import SimpleRAGPipeline

    rag = SimpleRAGPipeline(FakeVectorStore(), top_k=1)
    result = rag.ask("How long does standard shipping take?")

    assert "answer" in result
    assert "sources" in result
    assert "shipping_policy.md" in result["sources"]
    assert "3 to 5 business days" in result["answer"]


def test_rag_pipeline_low_similarity_returns_safe_answer():
    from src.rag_pipeline import SimpleRAGPipeline

    class LowScoreVectorStore:
        def search(self, query, top_k=3):
            return [{"source": "unknown.md", "text": "Unrelated content.", "similarity_score": 0.01}]

    rag = SimpleRAGPipeline(LowScoreVectorStore(), top_k=1, min_similarity_score=0.25)
    result = rag.ask("Can I pay with cryptocurrency?")

    assert "could not find enough information" in result["answer"].lower()
