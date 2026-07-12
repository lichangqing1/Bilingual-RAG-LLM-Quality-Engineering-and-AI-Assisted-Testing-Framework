from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_PATH = Path(__file__).resolve().parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

from src.document_loader import load_markdown_documents, validate_documents
from src.evaluator import (
    add_pass_fail_flags,
    evaluate_single_case,
    summarize_results,
)
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


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: str = ""
    expected_source: str = "none"
    expected_keywords: str = ""
    question_type: str = "normal"
    top_k: int = Field(3, ge=1, le=5)


class EvaluateResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    metrics: Dict[str, Any]
    latency_ms: float


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = ""
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
    expected_answer: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    log_path: str


def build_rag(top_k: int = 3) -> SimpleRAGPipeline:
    docs = load_markdown_documents(str(PROJECT_PATH / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = TfidfVectorStore()
    vector_store.build_index(chunks)
    return SimpleRAGPipeline(vector_store, top_k=top_k)


def json_safe_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    safe = {}
    for key, value in record.items():
        if pd.isna(value):
            safe[key] = None
        elif hasattr(value, "item"):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


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


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    start = time.perf_counter()
    try:
        rag = build_rag(top_k=request.top_k)
        result = rag.ask(request.question)
        expected = pd.Series({
            "question": request.question,
            "expected_answer": request.expected_answer,
            "expected_source": request.expected_source,
            "expected_keywords": request.expected_keywords,
            "question_type": request.question_type,
        })
        metrics = evaluate_single_case(result, expected)
        metrics_with_flags = json_safe_dict(add_pass_fail_flags(pd.DataFrame([metrics])).iloc[0].to_dict())
    except Exception as exc:
        append_jsonl(
            PROJECT_PATH / "logs" / "api_evaluations.jsonl",
            {
                "event": "api_evaluation_failed",
                "question": request.question,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response = EvaluateResponse(
        question=request.question,
        answer=str(result.get("answer", "")),
        sources=[str(source) for source in result.get("sources", [])],
        metrics=metrics_with_flags,
        latency_ms=latency_ms,
    )
    append_jsonl(
        PROJECT_PATH / "logs" / "api_evaluations.jsonl",
        {
            "event": "api_evaluation_completed",
            "question": response.question,
            "answer": response.answer,
            "sources": response.sources,
            "metrics": response.metrics,
            "latency_ms": response.latency_ms,
        },
    )
    return response


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    log_path = append_jsonl(
        PROJECT_PATH / "logs" / "feedback.jsonl",
        {
            "event": "feedback_received",
            "question": request.question,
            "answer": request.answer,
            "rating": request.rating,
            "comment": request.comment,
            "expected_answer": request.expected_answer,
        },
    )
    return FeedbackResponse(status="logged", log_path=log_path)


@app.get("/metrics")
def metrics() -> dict:
    summary_path = PROJECT_PATH / "results" / "summary_report.csv"
    if summary_path.exists():
        summary = json_safe_dict(pd.read_csv(summary_path).iloc[0].to_dict())
    else:
        summary = {}

    log_counts = {}
    for name, path in {
        "api_requests": PROJECT_PATH / "logs" / "api_requests.jsonl",
        "api_evaluations": PROJECT_PATH / "logs" / "api_evaluations.jsonl",
        "feedback": PROJECT_PATH / "logs" / "feedback.jsonl",
        "evaluation_runs": PROJECT_PATH / "logs" / "evaluation_runs.jsonl",
    }.items():
        if path.exists():
            log_counts[name] = len(path.read_text(encoding="utf-8").splitlines())
        else:
            log_counts[name] = 0

    return {
        "status": "ok",
        "summary": summary,
        "log_counts": log_counts,
    }
