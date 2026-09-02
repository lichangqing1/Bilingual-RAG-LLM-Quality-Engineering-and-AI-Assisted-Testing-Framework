from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ai_testing.schemas import GeneratedTestCase, TestScenario, model_to_dict


def save_scenarios(scenarios: Iterable[TestScenario], output_path: Path) -> str:
    """Save validated scenarios as reusable JSON test assets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [model_to_dict(scenario) for scenario in scenarios]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def save_cases_csv(cases: Iterable[GeneratedTestCase], output_path: Path) -> str:
    """Save generated cases in the same schema used by the evaluator."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [case.to_evaluation_row() for case in cases]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path)


def save_cases_json(cases: Iterable[GeneratedTestCase], output_path: Path) -> str:
    """Save full generated cases, including IDs and expected behavior, as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [model_to_dict(case) for case in cases]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def build_pytest_code(cases: List[GeneratedTestCase]) -> str:
    """
    Build pytest code from validated cases.

    Python owns the executable template; generated content only fills data.
    """
    rows = [case.to_evaluation_row() for case in cases]
    return "\n".join(
        [
            "import pandas as pd",
            "",
            "from src.evaluator import add_pass_fail_flags, evaluate_single_case",
            "",
            f"GENERATED_CASES = {repr(rows)}",
            "",
            "",
            "def test_ai_generated_cases_pass(rag_pipeline):",
            "    for row in GENERATED_CASES:",
            "        result = rag_pipeline.ask(row['question'])",
            "        metrics = evaluate_single_case(result, pd.Series(row))",
            "        flagged = add_pass_fail_flags(pd.DataFrame([metrics]))",
            "        assert flagged.iloc[0]['overall_pass'] == 1",
            "",
        ]
    )


def save_pytest_code(cases: List[GeneratedTestCase], output_path: Path) -> str:
    """Save generated pytest code after schema-based case validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_pytest_code(cases), encoding="utf-8")
    return str(output_path)
