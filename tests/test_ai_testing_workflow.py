import json

import pandas as pd

from ai_testing.case_generator import build_pytest_code, save_cases_csv, save_scenarios
from ai_testing.failure_analyzer import analyze_failed_cases
from ai_testing.generators import LLMTestGenerator, RuleBasedGenerator
from ai_testing.llm_client import OpenAICompatibleJSONClient, build_requirement_generation_prompt
from ai_testing.quality_summary import build_quality_summary
from ai_testing.requirement_parser import parse_requirement
from ai_testing.scenario_generator import generate_test_scenarios
from ai_testing.test_data_generator import generate_test_cases


def test_requirement_to_generated_cases_workflow(tmp_path):
    requirement = parse_requirement(
        """
        RAG answer quality
        - Answerable questions must cite a source.
        - Unsupported questions must be refused safely.
        - The suite must include bilingual English and Chinese examples.
        """,
        requirement_id="REQ-TEST-001",
    )

    scenarios = generate_test_scenarios(requirement)
    cases = generate_test_cases(scenarios)

    assert requirement.requirement_id == "REQ-TEST-001"
    assert "bilingual" in requirement.tags
    assert len(scenarios) >= 4
    assert {case.question_type for case in cases} >= {"normal", "unanswerable", "prompt_injection"}
    assert all(case.id for case in cases)
    assert all(case.query == case.question for case in cases)
    assert all(case.expected_behavior.required_keywords for case in cases)

    scenario_path = save_scenarios(scenarios, tmp_path / "scenarios.json")
    case_path = save_cases_csv(cases, tmp_path / "cases.csv")

    assert json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))[0]["requirement_id"] == "REQ-TEST-001"
    assert pd.read_csv(case_path).columns.tolist() == [
        "question",
        "expected_answer",
        "expected_source",
        "expected_keywords",
        "question_type",
    ]
    assert scenario_path.endswith("scenarios.json")


def test_generator_backends_are_schema_validated():
    requirement = parse_requirement("Account support must protect passwords.", requirement_id="REQ-GEN-001")
    rule_based = RuleBasedGenerator().generate(requirement)

    assert any(scenario.category == "sensitive_information_disclosure" for scenario in rule_based)

    class FakeLLMClient:
        def complete_json(self, prompt):
            return [
                {
                    "requirement_id": "REQ-GEN-001",
                    "name": "LLM generated account deletion scenario",
                    "type": "positive",
                    "category": "answerable_qa",
                    "language": "en",
                    "input": "How long can account deletion take?",
                    "expected_behavior": ["Answer must cite account_support.md."],
                }
            ]

    llm_generated = LLMTestGenerator(client=FakeLLMClient()).generate(requirement)

    assert llm_generated[0].requirement_id == "REQ-GEN-001"
    assert "Requirement ID: REQ-GEN-001" in build_requirement_generation_prompt(requirement)


def test_llm_generator_requires_configured_client():
    requirement = parse_requirement("RAG answers should be grounded.", requirement_id="REQ-GEN-002")

    try:
        LLMTestGenerator().generate(requirement)
    except RuntimeError as exc:
        assert "--generator rule_based" in str(exc)
        assert "LLM client" in str(exc)
    else:
        raise AssertionError("LLMTestGenerator should fail clearly when no client is configured.")


def test_openai_compatible_client_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("LLM_TIMEOUT", "7")

    client = OpenAICompatibleJSONClient.from_env()

    assert client.api_key == "test-key"
    assert client.model == "test-model"
    assert client.api_base == "https://example.test/v1"
    assert client.timeout == 7


def test_generated_pytest_code_uses_template_not_freeform_python():
    requirement = parse_requirement("RAG should cite sources and refuse unsupported questions.")
    cases = generate_test_cases(generate_test_scenarios(requirement))

    code = build_pytest_code(cases)

    assert "GENERATED_CASES" in code
    assert "evaluate_single_case" in code
    assert "rag_pipeline" in code
    assert "exec(" not in code


def test_failure_analysis_and_quality_summary_are_structured():
    failed_cases = pd.DataFrame(
        [
            {
                "question": "Can I pay with cryptocurrency?",
                "overall_pass": 0,
                "failure_type": "Safety failure: unanswerable question not handled safely",
                "failure_detail": "unanswerable question was not safely refused",
                "answer": "Yes, cryptocurrency is accepted.",
            }
        ]
    )
    analyses = analyze_failed_cases(failed_cases)

    assert analyses[0].failure_category == "UNSAFE_RESPONSE"
    assert analyses[0].likely_stage == "Safety Guard"
    assert "refusal rule too narrow" in analyses[0].possible_causes

    results = pd.DataFrame(
        [
            {"overall_pass": 1, "failure_type": "Passed"},
            {"overall_pass": 0, "failure_type": "Citation failure: answer source citation was missing or inaccurate"},
        ]
    )
    summary = build_quality_summary(results)

    assert summary.total_cases == 2
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.pass_rate == 0.5
    assert "Citation failure" in next(iter(summary.failure_category_counts))
