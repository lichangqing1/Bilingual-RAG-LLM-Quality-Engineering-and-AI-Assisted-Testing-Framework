from __future__ import annotations

import json
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

from ai_testing.requirement_parser import parse_requirement
from ai_testing.generators import RuleBasedGenerator
from ai_testing.schemas import model_to_dict
from ai_testing.test_data_generator import generate_test_cases
from src.document_loader import load_markdown_documents, validate_documents
from src.evaluation.evaluator import (
    add_pass_fail_flags,
    evaluate_single_case,
)
from src.logging_utils import append_jsonl
from src.rag_pipeline import SimpleRAGPipeline
from src.retrieval import build_vector_store
from src.text_splitter import create_chunks


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=5)
    retrieval_mode: str = "hybrid"
    semantic_backend: str = "local"


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
    retrieval_mode: str = "hybrid"
    semantic_backend: str = "local"


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


class TestingGenerateRequest(BaseModel):
    requirement_text: str = Field(..., min_length=1)
    requirement_id: str = "RAG-AI-001"


class TestingGenerateResponse(BaseModel):
    requirement: Dict[str, Any]
    scenarios: List[Dict[str, Any]]
    cases: List[Dict[str, Any]]


LOG_FILES = {
    "api_requests": "api_requests.jsonl",
    "api_evaluations": "api_evaluations.jsonl",
    "feedback": "feedback.jsonl",
    "evaluation_runs": "evaluation_runs.jsonl",
    "evaluation_failed_cases": "evaluation_failed_cases.jsonl",
    "security_evaluation_runs": "security_evaluation_runs.jsonl",
    "challenge_evaluation_runs": "challenge_evaluation_runs.jsonl",
    "ai_testing": "ai_testing.jsonl",
}
LOG_DIR = PROJECT_PATH / "logs"
LOG_PATH_OVERRIDES: Dict[str, Path] = {}


def write_log(log_name: str, record: Dict[str, Any]) -> str:
    """Write a log record and remember the concrete path used."""
    if log_name not in LOG_FILES:
        raise ValueError(f"Unknown log name: {log_name}")
    log_path = append_jsonl(LOG_DIR / LOG_FILES[log_name], record)
    LOG_PATH_OVERRIDES[log_name] = Path(log_path)
    return log_path


def get_log_path(log_name: str) -> Path:
    """Return the latest concrete path for a known log file."""
    return LOG_PATH_OVERRIDES.get(log_name, LOG_DIR / LOG_FILES[log_name])


def build_rag(top_k: int = 3, retrieval_mode: str = "hybrid", semantic_backend: str = "local") -> SimpleRAGPipeline:
    docs = load_markdown_documents(str(PROJECT_PATH / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = build_vector_store(retrieval_mode=retrieval_mode, semantic_backend=semantic_backend)
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


def summarize_jsonl_log(path: Path) -> Dict[str, Any]:
    """Return a compact, safe summary of a JSONL log file."""
    summary: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "line_count": 0,
        "last_timestamp": None,
        "last_event": None,
    }
    if not path.exists():
        return summary

    last_record: Dict[str, Any] = {}
    malformed_lines = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            summary["line_count"] += 1
            try:
                last_record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1

    summary["malformed_lines"] = malformed_lines
    if last_record:
        summary["last_timestamp"] = last_record.get("timestamp")
        summary["last_event"] = last_record.get("event")
    return summary


def build_logs_summary() -> Dict[str, Any]:
    """Summarize all known runtime and evaluation logs."""
    logs = {
        name: summarize_jsonl_log(get_log_path(name))
        for name in LOG_FILES
    }
    return {
        "status": "ok",
        "log_dir": str(LOG_DIR),
        "logs": logs,
        "total_log_lines": sum(item["line_count"] for item in logs.values()),
    }


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
        rag = build_rag(
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            semantic_backend=request.semantic_backend,
        )
        result = rag.ask(request.question)
    except Exception as exc:
        write_log(
            "api_requests",
            {
                "event": "api_request_failed",
                "question": request.question,
                "top_k": request.top_k,
                "retrieval_mode": request.retrieval_mode,
                "semantic_backend": request.semantic_backend,
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
    write_log(
        "api_requests",
        {
            "event": "api_request_completed",
            "question": response.question,
            "top_k": request.top_k,
            "retrieval_mode": request.retrieval_mode,
            "semantic_backend": request.semantic_backend,
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
        rag = build_rag(
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            semantic_backend=request.semantic_backend,
        )
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
        write_log(
            "api_evaluations",
            {
                "event": "api_evaluation_failed",
                "question": request.question,
                "retrieval_mode": request.retrieval_mode,
                "semantic_backend": request.semantic_backend,
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
    write_log(
        "api_evaluations",
        {
            "event": "api_evaluation_completed",
            "question": response.question,
            "answer": response.answer,
            "sources": response.sources,
            "retrieval_mode": request.retrieval_mode,
            "semantic_backend": request.semantic_backend,
            "metrics": response.metrics,
            "latency_ms": response.latency_ms,
        },
    )
    return response


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    log_path = write_log(
        "feedback",
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


@app.post("/testing/generate", response_model=TestingGenerateResponse)
def testing_generate(request: TestingGenerateRequest) -> TestingGenerateResponse:
    requirement = parse_requirement(
        request.requirement_text,
        requirement_id=request.requirement_id,
    )
    scenarios = RuleBasedGenerator().generate(requirement)
    cases = generate_test_cases(scenarios)
    response = TestingGenerateResponse(
        requirement=model_to_dict(requirement),
        scenarios=[model_to_dict(scenario) for scenario in scenarios],
        cases=[model_to_dict(case) for case in cases],
    )
    write_log(
        "ai_testing",
        {
            "event": "ai_testing_cases_generated",
            "requirement_id": requirement.requirement_id,
            "scenario_count": len(scenarios),
            "case_count": len(cases),
            "tags": requirement.tags,
        },
    )
    return response


@app.get("/metrics")
def metrics() -> dict:
    summary_path = PROJECT_PATH / "results" / "summary_report.csv"
    if summary_path.exists():
        summary = json_safe_dict(pd.read_csv(summary_path).iloc[0].to_dict())
    else:
        summary = {}

    logs_summary = build_logs_summary()
    log_counts = {
        name: info["line_count"]
        for name, info in logs_summary["logs"].items()
    }

    return {
        "status": "ok",
        "summary": summary,
        "log_counts": log_counts,
    }


@app.get("/logs/summary")
def logs_summary() -> dict:
    return build_logs_summary()
