"""Run the full RAG evaluation workflow from the command line."""
import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_markdown_documents, validate_documents
from src.text_splitter import create_chunks
from src.retrieval import build_vector_store
from src.rag_pipeline import SimpleRAGPipeline
from src.evaluator import evaluate_dataset, save_evaluation_outputs, identify_failed_cases, summarize_results
from src.logging_utils import append_jsonl
from src.report_generator import generate_markdown_report


def print_header(title: str) -> None:
    """Print a readable CLI section header."""
    line = "=" * 72
    print(f"\n{line}")
    print(title)
    print(line)


def print_subheader(title: str) -> None:
    """Print a readable CLI subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def print_key_values(values: Dict[str, object]) -> None:
    """Print aligned key/value pairs for command-line reports."""
    width = max((len(key) for key in values), default=0)
    for key, value in values.items():
        print(f"{key:<{width}} : {value}")


def format_metric(value: object) -> str:
    """Format summary metrics consistently."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_rag_eval_config(config_path: Path) -> Dict[str, object]:
    """
    Parse the small project YAML config without adding a YAML dependency.

    The parser intentionally reads only the fields needed by this runner:
    dataset.name, dataset.files.*, and retrieval.top_k.
    """
    config: Dict[str, object] = {
        "dataset_name": "bilingual_customer_support_rag_eval",
        "dataset_files": {
            "combined": "data/evaluation/evaluation_questions.csv",
            "english": "data/evaluation/rag_eval_en.csv",
            "chinese": "data/evaluation/rag_eval_zh.csv",
            "security": "data/evaluation/security_questions.csv",
        },
        "retrieval_mode": "hybrid",
        "semantic_backend": "local",
        "top_k": 3,
    }
    if not config_path.exists():
        return config

    section = ""
    subsection = ""
    dataset_files = dict(config["dataset_files"])
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()

        if indent == 0 and line.endswith(":"):
            section = line[:-1]
            subsection = ""
            continue

        if section == "dataset" and indent == 2:
            if line.startswith("name:"):
                config["dataset_name"] = line.split(":", 1)[1].strip().strip('"')
            elif line == "files:":
                subsection = "files"
            else:
                subsection = ""
            continue

        if section == "dataset" and subsection == "files" and indent == 4 and ":" in line:
            key, value = line.split(":", 1)
            dataset_files[key.strip()] = value.strip().strip('"')
            continue

        if section == "retrieval" and indent == 2 and line.startswith("top_k:"):
            config["top_k"] = int(line.split(":", 1)[1].strip())
        elif section == "retrieval" and indent == 2 and line.startswith("mode:"):
            config["retrieval_mode"] = line.split(":", 1)[1].strip().strip('"')
        elif section == "retrieval" and indent == 2 and line.startswith("semantic_backend:"):
            config["semantic_backend"] = line.split(":", 1)[1].strip().strip('"')

    config["dataset_files"] = dataset_files
    return config


def load_evaluation_dataset(config_path: Path | None = None) -> pd.DataFrame:
    """
    Load formal evaluation shards from config.

    The preferred path uses English, Chinese, and security CSV shards. If none
    of those files exist, the loader falls back to the legacy combined CSV.
    """
    config = parse_rag_eval_config(config_path or PROJECT_ROOT / "configs" / "rag_eval_config.yaml")
    dataset_files = config["dataset_files"]
    assert isinstance(dataset_files, dict)

    frames: List[pd.DataFrame] = []
    for split_name in ["english", "chinese", "security"]:
        relative_path = dataset_files.get(split_name)
        if not relative_path:
            continue
        path = PROJECT_ROOT / str(relative_path)
        if path.exists():
            frame = pd.read_csv(path)
            frame["dataset_split"] = split_name
            frames.append(frame)

    if frames:
        return pd.concat(frames, ignore_index=True)

    combined_path = PROJECT_ROOT / str(dataset_files.get("combined", "data/evaluation/evaluation_questions.csv"))
    evaluation_df = pd.read_csv(combined_path)
    evaluation_df["dataset_split"] = "combined"
    return evaluation_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation workflow.")
    parser.add_argument("--retrieval-mode", choices=["lexical", "keyword", "semantic", "hybrid"], default=None)
    parser.add_argument("--semantic-backend", choices=["local", "faiss"], default=None)
    args = parser.parse_args()

    config = parse_rag_eval_config(PROJECT_ROOT / "configs" / "rag_eval_config.yaml")
    if args.retrieval_mode:
        config["retrieval_mode"] = args.retrieval_mode
    if args.semantic_backend:
        config["semantic_backend"] = args.semantic_backend

    docs = load_markdown_documents(str(PROJECT_ROOT / "data" / "documents"))
    validate_documents(docs)
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    vector_store = build_vector_store(
        retrieval_mode=str(config.get("retrieval_mode", "hybrid")),
        semantic_backend=str(config.get("semantic_backend", "local")),
    )
    vector_store.build_index(chunks)
    rag = SimpleRAGPipeline(vector_store, top_k=int(config.get("top_k", 3)))

    evaluation_df = load_evaluation_dataset(PROJECT_ROOT / "configs" / "rag_eval_config.yaml")
    evaluation_results = evaluate_dataset(rag, evaluation_df)
    outputs = save_evaluation_outputs(evaluation_results, output_dir=str(PROJECT_ROOT / "results"))

    failed_cases = identify_failed_cases(evaluation_results)
    summary = summarize_results(evaluation_results)
    report_path = generate_markdown_report(summary, failed_cases, str(PROJECT_ROOT / "results" / "evaluation_report.md"))
    summary_record = summary.iloc[0].to_dict()
    append_jsonl(
        PROJECT_ROOT / "logs" / "evaluation_runs.jsonl",
        {
            "event": "evaluation_run_completed",
            "dataset_name": config.get("dataset_name", ""),
            "dataset_splits": sorted(evaluation_df.get("dataset_split", pd.Series(dtype=str)).dropna().unique().tolist()),
            "retrieval_mode": config.get("retrieval_mode", "hybrid"),
            "semantic_backend": config.get("semantic_backend", "local"),
            "total_cases": int(summary_record.get("total_questions", 0)),
            "answerable_cases": int(summary_record.get("answerable_questions", 0)),
            "unanswerable_cases": int(summary_record.get("unanswerable_questions", 0)),
            "security_cases": int(summary_record.get("security_questions", 0)),
            "overall_pass_rate": float(summary_record.get("overall_pass_rate", 0)),
            "avg_unanswerable_safe": float(summary_record.get("avg_unanswerable_safe", 0)),
            "failed_cases": int(len(failed_cases)),
            "report_path": report_path,
        },
    )
    for _, row in failed_cases.iterrows():
        append_jsonl(
            PROJECT_ROOT / "logs" / "evaluation_failed_cases.jsonl",
            {
                "event": "evaluation_failed_case",
                "question": row.get("question", ""),
                "language": row.get("language", ""),
                "question_type": row.get("question_type", ""),
                "failure_type": row.get("failure_type", ""),
                "failure_detail": row.get("failure_detail", ""),
                "recommendation": row.get("recommendation", ""),
            },
        )

    print_header("RAG Evaluation Completed")
    print_subheader("Run Configuration")
    print_key_values(
        {
            "Dataset": config.get("dataset_name", ""),
            "Dataset splits": ", ".join(
                sorted(evaluation_df.get("dataset_split", pd.Series(dtype=str)).dropna().unique().tolist())
            ),
            "Retrieval mode": config.get("retrieval_mode", "hybrid"),
            "Semantic backend": config.get("semantic_backend", "local"),
            "Top k": config.get("top_k", 3),
        }
    )

    print_subheader("Output Files")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"- markdown_report: {report_path}")

    print_subheader("Summary Metrics")
    summary_record = summary.iloc[0].to_dict()
    print_key_values({key: format_metric(value) for key, value in summary_record.items()})


if __name__ == "__main__":
    main()
