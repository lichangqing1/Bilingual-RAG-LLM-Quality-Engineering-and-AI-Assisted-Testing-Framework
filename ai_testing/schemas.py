from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ScenarioType = Literal["positive", "negative", "boundary", "security"]
FailureCategory = Literal[
    "RETRIEVAL_MISS",
    "RETRIEVAL_NOISE",
    "GROUNDING_FAILURE",
    "CITATION_FAILURE",
    "HALLUCINATION",
    "ABNORMAL_REFUSAL",
    "UNSAFE_RESPONSE",
    "ANSWER_RELEVANCE",
    "SYSTEM_FAILURE",
]


class RequirementSpec(BaseModel):
    """Structured representation of a feature or acceptance requirement."""

    requirement_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    acceptance_criteria: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class TestScenario(BaseModel):
    """AI-assisted test scenario generated from a requirement."""

    __test__ = False

    requirement_id: str
    name: str
    type: ScenarioType
    category: str
    language: Literal["en", "zh"] = "en"
    input: str
    expected_behavior: List[str] = Field(default_factory=list)


class ExpectedBehavior(BaseModel):
    """Machine-checkable expectation generated from a test scenario."""

    answerable: bool = True
    required_keywords: List[str] = Field(default_factory=list)
    expected_source: Optional[str] = None
    citation_required: bool = True
    safe_refusal_required: bool = False


class GeneratedTestCase(BaseModel):
    """Evaluation-ready test case created from a structured scenario."""

    id: str
    category: str
    query: str
    expected_behavior: ExpectedBehavior
    requirement_id: str
    scenario_name: str
    question: str
    expected_answer: str = ""
    expected_source: str = "none"
    expected_keywords: str = ""
    question_type: str = "normal"
    language: Literal["en", "zh"] = "en"

    def to_evaluation_row(self) -> Dict[str, str]:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_source": self.expected_source,
            "expected_keywords": self.expected_keywords,
            "question_type": self.question_type,
        }


class FailureAnalysis(BaseModel):
    """Structured failure triage output for a failed RAG case."""

    case_id: str = ""
    question: str
    failure_category: FailureCategory
    likely_stage: str
    evidence: str
    possible_causes: List[str]
    recommended_checks: List[str]


class LLMRootCauseAnalysis(BaseModel):
    """Optional LLM-assisted RCA output for a failed RAG case."""

    case_id: str
    failure_category: FailureCategory
    suspected_component: Literal[
        "retrieval",
        "generation",
        "grounding",
        "citation",
        "safety_guard",
        "evaluation",
        "test_data",
        "unknown",
    ]
    root_cause: str = Field(..., min_length=1)
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_actions: List[str] = Field(default_factory=list)


class QualitySummary(BaseModel):
    """Structured release-quality summary generated from evaluation outputs."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    failure_category_counts: Dict[str, int]
    summary: str


def model_to_dict(model: BaseModel) -> Dict[str, object]:
    """Return a pydantic model as a dict across pydantic v1/v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def model_validate(model_class, payload):
    """Validate a payload against a pydantic model across pydantic v1/v2."""
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(payload)
    return model_class.parse_obj(payload)
