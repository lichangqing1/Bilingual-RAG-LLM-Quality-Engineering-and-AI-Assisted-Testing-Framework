import pytest

from src.document_loader import load_markdown_documents, validate_documents
from src.rag_pipeline import SimpleRAGPipeline
from src.retrieval import build_vector_store
from src.text_splitter import create_chunks


@pytest.fixture(scope="session")
def rag_pipeline():
    """Shared deterministic RAG pipeline for generated-case tests."""
    docs = load_markdown_documents("data/documents")
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = build_vector_store(retrieval_mode="hybrid", semantic_backend="local")
    vector_store.build_index(chunks)
    return SimpleRAGPipeline(vector_store, top_k=3)
