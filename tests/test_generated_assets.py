from pathlib import Path

import pandas as pd
import pytest

from ai_testing.schemas import GeneratedTestCase
from src.evaluation.evaluator import add_pass_fail_flags, evaluate_single_case


GENERATED_CASES_PATH = Path("test_assets/generated_cases/ai_generated_cases.json")


def load_generated_cases():
    if not GENERATED_CASES_PATH.exists():
        return []
    raw_cases = pd.read_json(GENERATED_CASES_PATH).to_dict(orient="records")
    return [GeneratedTestCase(**case) for case in raw_cases]


@pytest.mark.parametrize("case", load_generated_cases(), ids=lambda case: case.id)
def test_generated_rag_cases_execute(case, rag_pipeline):
    result = rag_pipeline.ask(case.query)
    metrics = evaluate_single_case(result, pd.Series(case.to_evaluation_row()))
    flagged = add_pass_fail_flags(pd.DataFrame([metrics]))

    assert flagged.iloc[0]["overall_pass"] == 1
