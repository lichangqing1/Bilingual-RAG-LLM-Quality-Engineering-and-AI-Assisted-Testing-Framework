from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ai_testing.schemas import RequirementSpec


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class LLMClient(Protocol):
    """Minimal client interface for optional LLM-assisted testing."""

    def complete_json(self, prompt: str) -> object:
        """Return JSON-compatible output from a prompt."""


def load_prompt(name: str) -> str:
    """Load an AI-testing prompt template."""
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def build_requirement_generation_prompt(requirement: RequirementSpec) -> str:
    """Build a prompt for requirement-to-scenario generation."""
    template = load_prompt("test_generation.txt")
    return template.format(
        requirement_id=requirement.requirement_id,
        title=requirement.title,
        description=requirement.description,
        acceptance_criteria="\n".join(f"- {item}" for item in requirement.acceptance_criteria),
        tags=", ".join(requirement.tags),
    )


def parse_json_response(response: object) -> object:
    """Parse a JSON string or pass through already-decoded JSON-compatible data."""
    if isinstance(response, str):
        return json.loads(response)
    return response
