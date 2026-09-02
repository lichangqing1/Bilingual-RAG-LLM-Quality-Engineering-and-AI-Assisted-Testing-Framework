from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ai_testing.schemas import GeneratedTestCase
from src.document_loader import load_markdown_documents, validate_documents
from src.evaluation.evaluator import add_pass_fail_flags, evaluate_dataset, identify_failed_cases
from src.rag_pipeline import SimpleRAGPipeline
from src.retrieval import build_vector_store
from src.text_splitter import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_generated_case_rag(
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
    top_k: int = 3,
) -> SimpleRAGPipeline:
    """Build the deterministic RAG pipeline used for generated-case execution."""
    docs = load_markdown_documents(str(PROJECT_ROOT / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)
    vector_store = build_vector_store(retrieval_mode=retrieval_mode, semantic_backend=semantic_backend)
    vector_store.build_index(chunks)
    return SimpleRAGPipeline(vector_store, top_k=top_k)


def cases_to_dataframe(cases: Iterable[GeneratedTestCase]) -> pd.DataFrame:
    """Convert generated test cases to the evaluator CSV schema."""
    return pd.DataFrame([case.to_evaluation_row() for case in cases])


def execute_generated_cases(
    cases: List[GeneratedTestCase],
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run generated cases through the real RAG pipeline and return results plus failures."""
    rag = build_generated_case_rag(
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
        top_k=top_k,
    )
    evaluation_df = cases_to_dataframe(cases)
    results = add_pass_fail_flags(evaluate_dataset(rag, evaluation_df))
    id_lookup = {case.question: case.id for case in cases}
    category_lookup = {case.question: case.category for case in cases}
    results["case_id"] = results["question"].map(id_lookup)
    results["generated_category"] = results["question"].map(category_lookup)
    failed_cases = identify_failed_cases(results)
    return results, failed_cases
