"""Run the AI-assisted testing workflow for requirement-to-test generation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_testing.case_generator import save_cases_csv, save_cases_json, save_pytest_code, save_scenarios
from ai_testing.executor import execute_generated_cases
from ai_testing.failure_analyzer import analyze_failed_cases
from ai_testing.generators import LLMTestGenerator, RuleBasedGenerator
from ai_testing.llm_client import OpenAICompatibleJSONClient
from ai_testing.quality_summary import build_quality_summary
from ai_testing.rca_analyzer import analyze_failed_cases_with_llm
from ai_testing.requirement_parser import parse_requirement
from ai_testing.schemas import model_to_dict
from ai_testing.test_data_generator import generate_test_cases


DEFAULT_REQUIREMENT = PROJECT_ROOT / "test_assets" / "requirements" / "sample_rag_requirement.md"
GENERATED_DIR = PROJECT_ROOT / "test_assets" / "generated_cases"
RESULTS_DIR = PROJECT_ROOT / "results"


def print_header(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}")
    print(title)
    print(line)


def print_subheader(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def write_json(payload: object, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run requirement-to-test AI-assisted workflow.")
    parser.add_argument("--requirement-file", default=str(DEFAULT_REQUIREMENT))
    parser.add_argument("--requirement-id", default="RAG-AI-001")
    parser.add_argument("--generator", choices=["rule_based", "llm"], default="rule_based")
    parser.add_argument("--rca", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--retrieval-mode", choices=["lexical", "keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--semantic-backend", choices=["local", "faiss"], default="local")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    requirement_path = Path(args.requirement_file)
    requirement_text = requirement_path.read_text(encoding="utf-8")
    requirement = parse_requirement(requirement_text, requirement_id=args.requirement_id)
    if args.generator == "llm":
        generator = LLMTestGenerator(client=OpenAICompatibleJSONClient.from_env())
    else:
        generator = RuleBasedGenerator()
    scenarios = generator.generate(requirement)
    cases = generate_test_cases(scenarios)

    scenario_path = save_scenarios(scenarios, GENERATED_DIR / "ai_testing_scenarios.json")
    case_path = save_cases_csv(cases, GENERATED_DIR / "ai_generated_eval_cases.csv")
    case_json_path = save_cases_json(cases, GENERATED_DIR / "ai_generated_cases.json")
    pytest_path = save_pytest_code(cases, GENERATED_DIR / "generated_pytest_cases.py")

    generated_results, failed_cases = execute_generated_cases(
        cases,
        retrieval_mode=args.retrieval_mode,
        semantic_backend=args.semantic_backend,
        top_k=args.top_k,
    )
    generated_results_path = GENERATED_DIR / "ai_generated_evaluation_results.csv"
    failed_cases_path = GENERATED_DIR / "ai_generated_failed_cases.csv"
    generated_results.to_csv(generated_results_path, index=False)
    failed_cases.to_csv(failed_cases_path, index=False)

    quality_summary = build_quality_summary(generated_results)
    failure_analysis = analyze_failed_cases(failed_cases)
    quality_summary_path = write_json(model_to_dict(quality_summary), GENERATED_DIR / "quality_summary.json")
    failure_analysis_path = write_json(
        [model_to_dict(item) for item in failure_analysis],
        GENERATED_DIR / "failure_analysis.json",
    )
    llm_rca = []
    if args.rca == "llm" and not failed_cases.empty:
        llm_rca = analyze_failed_cases_with_llm(
            failed_cases,
            failure_analysis,
            OpenAICompatibleJSONClient.from_env(),
        )
    llm_rca_path = write_json(
        [model_to_dict(item) for item in llm_rca],
        GENERATED_DIR / "llm_rca_analysis.json",
    )

    print_header("AI-Assisted Testing Workflow Completed")
    print_subheader("Requirement")
    print(f"- requirement_id: {requirement.requirement_id}")
    print(f"- requirement_file: {requirement_path}")
    print(f"- title: {requirement.title}")
    print(f"- tags: {', '.join(requirement.tags)}")
    print(f"- generator: {args.generator}")
    print(f"- rca: {args.rca}")

    print_subheader("Generated Test Assets")
    print(f"- scenarios: {scenario_path}")
    print(f"- generated_cases_json: {case_json_path}")
    print(f"- evaluation_cases: {case_path}")
    print(f"- pytest_template: {pytest_path}")
    print(f"- generated_evaluation_results: {generated_results_path}")
    print(f"- generated_failed_cases: {failed_cases_path}")
    print(f"- quality_summary: {quality_summary_path}")
    print(f"- failure_analysis: {failure_analysis_path}")
    print(f"- llm_rca_analysis: {llm_rca_path}")

    print_subheader("Counts")
    print(f"- scenarios: {len(scenarios)}")
    print(f"- generated_cases: {len(cases)}")
    print(f"- generated_case_failures: {len(failed_cases)}")
    print(f"- generated_case_pass_rate: {quality_summary.pass_rate:.4f}")
    print(f"- llm_rca_items: {len(llm_rca)}")


if __name__ == "__main__":
    main()
