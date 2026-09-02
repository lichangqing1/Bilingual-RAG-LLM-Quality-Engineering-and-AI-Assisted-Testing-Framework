from __future__ import annotations

from typing import Protocol

from ai_testing.schemas import RequirementSpec, TestScenario
from ai_testing.scenario_generator import generate_test_scenarios
from ai_testing.llm_client import build_requirement_generation_prompt, parse_json_response


class TestGenerator(Protocol):
    """Interface for requirement-driven test scenario generators."""

    def generate(self, requirement: RequirementSpec) -> list[TestScenario]:
        """Generate validated scenarios from a structured requirement."""


class RuleBasedGenerator:
    """Deterministic generator used by local runs and CI."""

    def generate(self, requirement: RequirementSpec) -> list[TestScenario]:
        return generate_test_scenarios(requirement)


class LLMTestGenerator:
    """
    Optional LLM generation adapter.

    The adapter expects a client object with `generate_scenarios(requirement)`.
    Returned payloads are converted into TestScenario objects so the same
    schema validation protects the downstream workflow.
    """

    def __init__(self, client=None, fallback: TestGenerator | None = None):
        self.client = client
        self.fallback = fallback or RuleBasedGenerator()

    def generate(self, requirement: RequirementSpec) -> list[TestScenario]:
        if self.client is None:
            return self.fallback.generate(requirement)
        if hasattr(self.client, "generate_scenarios"):
            payload = self.client.generate_scenarios(requirement)
        else:
            prompt = build_requirement_generation_prompt(requirement)
            payload = parse_json_response(self.client.complete_json(prompt))
        return [item if isinstance(item, TestScenario) else TestScenario(**item) for item in payload]
