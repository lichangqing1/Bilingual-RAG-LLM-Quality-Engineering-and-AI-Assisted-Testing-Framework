from __future__ import annotations

from typing import List

import pandas as pd

from ai_testing.schemas import FailureAnalysis


def classify_failure_category(failure_type: str) -> str:
    """Map evaluator failure text to a stable AI-testing taxonomy."""
    if "expected source not retrieved" in failure_type or "Retrieval failure" in failure_type:
        return "RETRIEVAL_MISS"
    if "expected keywords missing from retrieved context" in failure_type:
        return "RETRIEVAL_NOISE"
    if "Citation failure" in failure_type:
        return "CITATION_FAILURE"
    if "Faithfulness failure" in failure_type:
        return "GROUNDING_FAILURE"
    if "expected keywords missing" in failure_type or "Answer failure" in failure_type:
        return "ANSWER_RELEVANCE"
    if "Safety failure" in failure_type:
        return "UNSAFE_RESPONSE"
    if "hallucination" in failure_type.lower():
        return "HALLUCINATION"
    return "SYSTEM_FAILURE"


def _possible_causes(failure_type: str) -> List[str]:
    if "Retrieval failure" in failure_type:
        return ["missing knowledge-base coverage", "chunking mismatch", "retriever ranking issue"]
    if "Citation failure" in failure_type:
        return ["citation mapping error", "answer template omitted source", "retrieved source not propagated"]
    if "Faithfulness failure" in failure_type:
        return ["answer includes unsupported claims", "generation used weak evidence", "grounding guard too permissive"]
    if "Safety failure" in failure_type:
        return ["unsafe prompt pattern missing", "refusal rule too narrow", "unsupported intent not detected"]
    return ["metric threshold mismatch", "expected labels need review", "case requires manual inspection"]


def _recommended_checks(failure_type: str) -> List[str]:
    if "Retrieval failure" in failure_type:
        return ["inspect retrieved_sources", "compare expected_source with top-k chunks", "review lexical and semantic scores"]
    if "Citation failure" in failure_type:
        return ["check Source: filename formatting", "verify citation appears in retrieved_sources"]
    if "Faithfulness failure" in failure_type:
        return ["compare answer claims with retrieved_context", "tighten extractive answer selection"]
    if "Safety failure" in failure_type:
        return ["add prompt pattern to safety guard", "add bilingual refusal phrase coverage"]
    return ["review failed row metrics", "inspect answer and expected keywords"]


def analyze_failed_cases(failed_cases: pd.DataFrame) -> List[FailureAnalysis]:
    """Create structured failure triage and RCA suggestions for failed rows."""
    analyses: List[FailureAnalysis] = []
    for _, row in failed_cases.iterrows():
        failure_type = str(row.get("failure_type", "Other failure"))
        analyses.append(
            FailureAnalysis(
                case_id=str(row.get("case_id", "")),
                question=str(row.get("question", "")),
                failure_category=classify_failure_category(failure_type),
                likely_stage=_likely_stage(failure_type),
                evidence=str(row.get("failure_detail", "")) or str(row.get("answer", "")),
                possible_causes=_possible_causes(failure_type),
                recommended_checks=_recommended_checks(failure_type),
            )
        )
    return analyses


def _likely_stage(failure_type: str) -> str:
    if "Retrieval failure" in failure_type:
        return "Retrieval"
    if "Citation failure" in failure_type:
        return "Generation"
    if "Faithfulness failure" in failure_type:
        return "Generation/Grounding"
    if "Safety failure" in failure_type:
        return "Safety Guard"
    return "Evaluation"
