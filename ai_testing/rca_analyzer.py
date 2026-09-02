from __future__ import annotations

import json
from typing import Iterable, List

import pandas as pd

from ai_testing.llm_client import LLMClient, load_prompt, parse_json_response
from ai_testing.schemas import FailureAnalysis, LLMRootCauseAnalysis, model_validate


METRIC_FIELDS = [
    "source_match",
    "keyword_recall",
    "context_keyword_recall",
    "answer_groundedness",
    "hallucination_risk",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "citation_accuracy",
    "unanswerable_safe",
    "overall_pass",
]


def build_metric_summary(row: pd.Series) -> str:
    """Build a compact JSON metric summary for LLM RCA prompts."""
    metrics = {}
    for field in METRIC_FIELDS:
        value = row.get(field, None)
        if pd.isna(value):
            continue
        if isinstance(value, (int, float)):
            metrics[field] = round(float(value), 4)
        else:
            metrics[field] = value
    return json.dumps(metrics, ensure_ascii=False, sort_keys=True)


def analyze_failure_with_llm(
    row: pd.Series,
    deterministic_analysis: FailureAnalysis,
    client: LLMClient,
) -> LLMRootCauseAnalysis:
    """Enrich one deterministic failure analysis with evidence-grounded LLM RCA."""
    prompt_template = load_prompt("failure_analysis.txt")
    case_id = deterministic_analysis.case_id or str(row.get("case_id", "")) or "unknown"
    prompt = prompt_template.format(
        case_id=case_id,
        question=str(row.get("question", "")),
        expected_answer=str(row.get("expected_answer", "")),
        expected_source=str(row.get("expected_source", "")),
        expected_keywords=str(row.get("expected_keywords", "")),
        answer=str(row.get("answer", "")),
        retrieved_sources=str(row.get("retrieved_sources", "")),
        failure_type=str(row.get("failure_type", "")),
        failure_category=deterministic_analysis.failure_category,
        failure_detail=str(row.get("failure_detail", "")),
        metrics=build_metric_summary(row),
    )
    payload = parse_json_response(client.complete_json(prompt))
    return model_validate(LLMRootCauseAnalysis, payload)


def analyze_failed_cases_with_llm(
    failed_cases: pd.DataFrame,
    deterministic_analyses: Iterable[FailureAnalysis],
    client: LLMClient,
) -> List[LLMRootCauseAnalysis]:
    """Run optional LLM RCA for failed cases after deterministic classification."""
    if failed_cases.empty:
        return []

    analyses = list(deterministic_analyses)
    results: List[LLMRootCauseAnalysis] = []
    for index, (_, row) in enumerate(failed_cases.iterrows()):
        deterministic_analysis = analyses[index]
        results.append(analyze_failure_with_llm(row, deterministic_analysis, client))
    return results
