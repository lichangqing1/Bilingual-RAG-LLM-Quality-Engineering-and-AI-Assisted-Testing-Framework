from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_framework_files_exist():
    expected_files = [
        "configs/rag_eval_config.yaml",
        "requirements-vector.txt",
        "docs/ARCHITECTURE.md",
        "docs/EVALUATION_METRICS.md",
        "docs/SECURITY_EVALUATION.md",
        "docs/PROJECT_PORTFOLIO_CN.md",
        "scripts/run_security_evaluation.py",
        "data/evaluation/rag_eval_en.csv",
        "data/evaluation/rag_eval_zh.csv",
        "data/evaluation/security_questions.csv",
    ]

    for relative_path in expected_files:
        assert (PROJECT_ROOT / relative_path).exists()


def test_split_eval_files_keep_base_schema():
    base_columns = list(pd.read_csv(PROJECT_ROOT / "data/evaluation/evaluation_questions.csv", nrows=0).columns)

    for filename in ["rag_eval_en.csv", "rag_eval_zh.csv", "security_questions.csv"]:
        columns = list(pd.read_csv(PROJECT_ROOT / "data/evaluation" / filename, nrows=0).columns)
        assert columns == base_columns


def test_config_declares_core_tasks_and_metrics():
    config_text = (PROJECT_ROOT / "configs/rag_eval_config.yaml").read_text(encoding="utf-8")

    for expected in [
        "bilingual_customer_support_rag_eval",
        "answerable_qa",
        "unanswerable_qa",
        "hallucination_check",
        "prompt_injection",
        "jailbreak",
        "system_prompt_leakage",
        "sensitive_information_disclosure",
        "retrieval_poisoning",
        "unsafe_instruction_refusal",
        "source_grounding",
        "retrieval_relevance",
        "answer_groundedness",
        "source_citation_rate",
        "unanswerable_safe_rate",
        "hallucination_risk",
        "security_pass_rate",
    ]:
        assert expected in config_text


def test_vector_dependencies_are_optional():
    default_requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    vector_requirements = (PROJECT_ROOT / "requirements-vector.txt").read_text(encoding="utf-8")

    assert "faiss-cpu" not in default_requirements
    assert "sentence-transformers" not in default_requirements
    assert "faiss-cpu" in vector_requirements
    assert "sentence-transformers" in vector_requirements


def test_run_evaluation_loads_formal_dataset_shards():
    from scripts.run_evaluation import load_evaluation_dataset

    evaluation_df = load_evaluation_dataset(PROJECT_ROOT / "configs/rag_eval_config.yaml")

    assert set(evaluation_df["dataset_split"]) == {"english", "chinese", "security"}
    assert len(evaluation_df[evaluation_df["dataset_split"] == "english"]) == 32
    assert len(evaluation_df[evaluation_df["dataset_split"] == "chinese"]) == 29
    assert len(evaluation_df[evaluation_df["dataset_split"] == "security"]) == 12
    assert "prompt_injection" in set(evaluation_df["question_type"])
