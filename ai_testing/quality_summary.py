from __future__ import annotations

from typing import Dict

import pandas as pd

from ai_testing.schemas import QualitySummary


def build_quality_summary(evaluation_results: pd.DataFrame) -> QualitySummary:
    """
    Build an AI-style quality summary from metrics computed by code.

    The numbers come from deterministic evaluation. The narrative explains the
    release quality in plain language without changing the measured results.
    """
    if evaluation_results.empty:
        return QualitySummary(
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            pass_rate=0.0,
            failure_category_counts={},
            summary="No evaluation cases were available for this run.",
        )

    if "overall_pass" not in evaluation_results.columns:
        raise ValueError("evaluation_results must include overall_pass. Run add_pass_fail_flags first.")

    total = int(len(evaluation_results))
    passed = int((evaluation_results["overall_pass"] == 1).sum())
    failed = total - passed
    pass_rate = passed / total if total else 0.0
    if "failure_type" in evaluation_results.columns:
        failure_counts: Dict[str, int] = {
            str(key): int(value)
            for key, value in evaluation_results[evaluation_results["overall_pass"] == 0]["failure_type"].value_counts().items()
        }
    else:
        failure_counts = {}

    if failed == 0:
        narrative = (
            f"All {total} prepared cases passed. The current run shows no regression "
            "across retrieval, grounding, citation, refusal, or safety checks."
        )
    else:
        major_area = max(failure_counts, key=failure_counts.get) if failure_counts else "uncategorized failures"
        narrative = (
            f"{passed} of {total} cases passed. The main regression area is {major_area}; "
            "review the failed-case analysis before release."
        )

    return QualitySummary(
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        pass_rate=pass_rate,
        failure_category_counts=failure_counts,
        summary=narrative,
    )
