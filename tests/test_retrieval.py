from src.document_loader import load_markdown_documents
from src.tfidf_vector_store import TfidfVectorStore
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
