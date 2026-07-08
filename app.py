from pathlib import Path
import sys

import streamlit as st

PROJECT_PATH = Path(__file__).resolve().parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

from src.document_loader import load_markdown_documents, validate_documents
from src.text_splitter import create_chunks
from src.tfidf_vector_store import TfidfVectorStore
from src.rag_pipeline import SimpleRAGPipeline


st.set_page_config(page_title="Customer Support RAG Assistant", layout="wide")
st.title("AI Customer Support RAG Assistant")
st.caption("RAG chatbot with source-grounded answers and evaluation-focused design.")

DOCUMENT_DIR = PROJECT_PATH / "data" / "documents"


@st.cache_resource
def load_rag_pipeline(chunk_size: int, overlap: int, top_k: int):
    docs = load_markdown_documents(str(DOCUMENT_DIR))
    validate_documents(docs)

    chunks = create_chunks(docs, chunk_size=chunk_size, overlap=overlap)

    vector_store = TfidfVectorStore()
    vector_store.build_index(chunks)

    rag = SimpleRAGPipeline(vector_store, top_k=top_k)
    return rag, docs, chunks


with st.sidebar:
    st.header("RAG Settings")
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1000, value=500, step=100)
    overlap = st.slider("Overlap", min_value=0, max_value=300, value=100, step=50)
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=5, value=3, step=1)

    st.header("Example Questions")
    example_question = st.selectbox(
        "Choose an example",
        [
            "How long does standard shipping usually take?",
            "Can customized products be returned?",
            "What payment methods does the company accept?",
            "Can customer support see my password?",
            "Can I pay with cryptocurrency?",
        ],
    )

rag, docs, chunks = load_rag_pipeline(chunk_size, overlap, top_k)

st.subheader("Knowledge Base Summary")
col1, col2 = st.columns(2)
col1.metric("Documents", len(docs))
col2.metric("Chunks", len(chunks))

question = st.text_input("Ask a customer support question:", value=example_question)

if st.button("Ask") and question.strip():
    response = rag.ask(question)

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Retrieved Sources")
    for i, chunk in enumerate(response["retrieved_chunks"], start=1):
        with st.expander(f"Retrieved Chunk {i}: {chunk['source']} | Score: {chunk.get('similarity_score', 0):.4f}"):
            st.write(chunk["text"])

st.divider()
st.subheader("Evaluation Design")
st.write(
    "This project includes automated checks for source matching, keyword recall, "
    "safe handling of unanswerable questions, failed-case extraction, and experiment comparison."
)
