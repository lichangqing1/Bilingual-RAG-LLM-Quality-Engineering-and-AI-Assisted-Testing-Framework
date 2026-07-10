from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_PATH = Path(__file__).resolve().parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

from src.document_loader import load_markdown_documents, validate_documents
from src.logging_utils import append_jsonl
from src.rag_pipeline import SimpleRAGPipeline
from src.text_splitter import create_chunks
from src.tfidf_vector_store import TfidfVectorStore


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=5)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieved_context: str
    latency_ms: float


def build_rag(top_k: int = 3) -> SimpleRAGPipeline:
    docs = load_markdown_documents(str(PROJECT_PATH / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = TfidfVectorStore()
    vector_store.build_index(chunks)
    return SimpleRAGPipeline(vector_store, top_k=top_k)


app = FastAPI(
    title="Bilingual RAG Evaluation API",
    description="FastAPI endpoint for the bilingual RAG assistant with request logging.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()
    try:
        rag = build_rag(top_k=request.top_k)
        result = rag.ask(request.question)
    except Exception as exc:
        append_jsonl(
            PROJECT_PATH / "logs" / "api_requests.jsonl",
            {
                "event": "api_request_failed",
                "question": request.question,
                "top_k": request.top_k,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response = AskResponse(
        question=request.question,
        answer=str(result.get("answer", "")),
        sources=[str(source) for source in result.get("sources", [])],
        retrieved_context=str(result.get("retrieved_context", "")),
        latency_ms=latency_ms,
    )
    append_jsonl(
        PROJECT_PATH / "logs" / "api_requests.jsonl",
        {
            "event": "api_request_completed",
            "question": response.question,
            "top_k": request.top_k,
            "answer": response.answer,
            "sources": response.sources,
            "latency_ms": response.latency_ms,
        },
    )
    return response
