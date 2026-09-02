from __future__ import annotations

import re
from typing import List

from ai_testing.schemas import RequirementSpec


def _extract_bullets(text: str) -> List[str]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped):
            bullets.append(re.sub(r"^[-*]\s+", "", stripped).strip())
    return bullets


def _infer_tags(text: str) -> List[str]:
    normalized = text.lower()
    tag_rules = {
        "retrieval": ["retrieve", "retrieval", "context", "source", "evidence"],
        "citation": ["cite", "citation", "source"],
        "faithfulness": ["grounded", "faithful", "supported", "hallucination"],
        "refusal": ["refuse", "unsupported", "unanswerable", "not mention"],
        "security": ["prompt injection", "jailbreak", "system prompt", "sensitive"],
        "bilingual": ["chinese", "english", "bilingual", "中文", "英文"],
    }
    tags = [
        tag
        for tag, needles in tag_rules.items()
        if any(needle in normalized or needle in text for needle in needles)
    ]
    return tags or ["rag_quality"]


def parse_requirement(text: str, requirement_id: str = "RAG-AI-001") -> RequirementSpec:
    """Convert free-form requirement text into a structured testable spec."""
    if not text or not text.strip():
        raise ValueError("Requirement text cannot be empty.")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0].strip("# ") if lines else "RAG quality requirement"
    acceptance_criteria = _extract_bullets(text)
    if not acceptance_criteria:
        acceptance_criteria = [
            "Answer must be grounded in retrieved context.",
            "Answer must cite a relevant source when the question is answerable.",
            "Unsupported questions must be refused safely.",
        ]

    return RequirementSpec(
        requirement_id=requirement_id,
        title=title,
        description=text.strip(),
        acceptance_criteria=acceptance_criteria,
        tags=_infer_tags(text),
    )
