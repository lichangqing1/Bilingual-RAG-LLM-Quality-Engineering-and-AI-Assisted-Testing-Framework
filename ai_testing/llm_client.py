from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from urllib import request

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


class OpenAICompatibleJSONClient:
    """
    Minimal OpenAI-compatible chat-completions client for optional LLM generation.

    This is intentionally tiny and dependency-free. The default project path
    remains deterministic; this client is only used when explicitly configured.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM generator requested but no API key is configured. "
                "Set LLM_API_KEY or OPENAI_API_KEY, or use --generator rule_based."
            )
        return cls(
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )

    def complete_json(self, prompt: str) -> object:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not include Markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        return json.loads(content)
