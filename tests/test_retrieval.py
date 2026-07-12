from src.document_loader import load_markdown_documents
from src.tfidf_vector_store import HybridVectorStore, TfidfVectorStore
from src.text_splitter import create_chunks


def test_retrieval_returns_expected_structure():
    docs = load_markdown_documents("data/documents")
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    vector_store = TfidfVectorStore()
    vector_store.build_index(chunks)

    results = vector_store.search("How long does standard shipping take?", top_k=3)

    assert len(results) > 0
    assert "source" in results[0]
    assert "text" in results[0]
    assert "similarity_score" in results[0]


def test_hybrid_retrieval_exposes_bm25_and_embedding_scores():
    docs = load_markdown_documents("data/documents")
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    vector_store = HybridVectorStore()
    vector_store.build_index(chunks)

    results = vector_store.search("公司支持支付宝付款吗?", top_k=3)

    assert len(results) == 3
    assert results[0]["retrieval_method"] == "hybrid_bm25_embeddings"
    assert "bm25_score" in results[0]
    assert "embedding_score" in results[0]
    assert "similarity_score" in results[0]
