from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def normalize_text(text: object) -> str:
    """Normalize text for rule-based evaluation."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"source:\s*[^.]+\.?", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_expected_keywords(expected_keywords: object) -> List[str]:
    """Parse semicolon-separated expected keywords."""
    if pd.isna(expected_keywords):
        return []
    return [kw.strip() for kw in str(expected_keywords).split(";") if kw.strip()]


def keyword_recall(answer: object, expected_keywords: object) -> Optional[float]:
    """Calculate proportion of expected keywords found in the answer."""
    keywords = parse_expected_keywords(expected_keywords)
    if not keywords:
        return None
    answer_norm = normalize_text(answer)
    matched = [kw for kw in keywords if normalize_text(kw) in answer_norm]
    return len(matched) / len(keywords)


def context_keyword_recall(retrieved_context: object, expected_keywords: object) -> Optional[float]:
    """Calculate how many expected keywords appear in the retrieved context."""
    keywords = parse_expected_keywords(expected_keywords)
    if not keywords:
        return None
    context_norm = normalize_text(retrieved_context)
    matched = [kw for kw in keywords if normalize_text(kw) in context_norm]
    return len(matched) / len(keywords)


def matched_keywords(answer: object, expected_keywords: object) -> List[str]:
    """Return expected keywords that appear in the generated answer."""
    keywords = parse_expected_keywords(expected_keywords)
    answer_norm = normalize_text(answer)
    return [kw for kw in keywords if normalize_text(kw) in answer_norm]


def missing_keywords(answer: object, expected_keywords: object) -> List[str]:
    """Return expected keywords that do not appear in the generated answer."""
    keywords = parse_expected_keywords(expected_keywords)
    answer_norm = normalize_text(answer)
    return [kw for kw in keywords if normalize_text(kw) not in answer_norm]


def source_match(retrieved_sources: Iterable[str], expected_source: object) -> Optional[int]:
    """Check whether the expected source appears in retrieved sources."""
    if pd.isna(expected_source):
        return None
    expected_source = str(expected_source).strip()
    if expected_source.lower() == "none":
        return None
    retrieved_sources = [str(src).strip() for src in retrieved_sources]
    return int(expected_source in retrieved_sources)


def answerability_check(answer: object, question_type: object) -> Optional[int]:
    """Check whether unanswerable questions are handled safely."""
    if pd.isna(question_type):
        return None
    if str(question_type).lower().strip() != "unanswerable":
        return None
    answer_norm = normalize_text(answer)
    safe_phrases = [
        "could not find", "do not provide", "does not provide",
        "do not mention", "does not mention", "not mention",
        "not provided", "cannot confirm", "not enough information",
        "answer this safely", "insufficient information"
    ]
    return int(any(phrase in answer_norm for phrase in safe_phrases))


def answer_has_source_citation(answer: object) -> int:
    """Check whether an answer explicitly includes a source citation."""
    return int("source:" in str(answer).lower())


def _sentence_overlap_ratio(sentence: str, context: str) -> float:
    sentence_terms = set(re.findall(r"[a-zA-Z0-9$]+", normalize_text(sentence)))
    context_terms = set(re.findall(r"[a-zA-Z0-9$]+", normalize_text(context)))
    if not sentence_terms:
        return 1.0
    return len(sentence_terms.intersection(context_terms)) / len(sentence_terms)


def answer_groundedness(answer: object, retrieved_context: object, question_type: object = "normal") -> Optional[float]:
    """
    Estimate whether generated answer claims are grounded in retrieved context.

    This is a lightweight proxy suitable for a portfolio project. It rewards
    extractive answers whose claim sentences are present in or strongly overlap
    with retrieved evidence. Safe refusals for unanswerable questions are not
    penalized and return None.
    """
    if str(question_type).lower().strip() == "unanswerable":
        return None

    answer_norm = normalize_text(answer)
    context_norm = normalize_text(retrieved_context)
    if not answer_norm:
        return 0.0
    if answer_norm in context_norm:
        return 1.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(answer)) if s.strip()]
    claim_sentences = [s for s in sentences if not s.lower().startswith("source:")]
    if not claim_sentences:
        return 0.0

    scores = []
    for sentence in claim_sentences:
        sentence_norm = normalize_text(sentence)
        if not sentence_norm:
            continue
        if sentence_norm in context_norm:
            scores.append(1.0)
        else:
            scores.append(_sentence_overlap_ratio(sentence_norm, context_norm))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def hallucination_risk(answer: object, retrieved_context: object, question_type: object = "normal") -> Optional[float]:
    """Return a simple hallucination-risk proxy: 1 - answer_groundedness."""
    groundedness = answer_groundedness(answer, retrieved_context, question_type)
    if groundedness is None:
        return None
    return 1 - groundedness


def evaluate_single_case(result: Dict[str, object], expected_row: pd.Series) -> Dict[str, object]:
    """Evaluate one RAG result against one expected evaluation row."""
    answer = result.get("answer", "")
    retrieved_sources = result.get("sources", [])
    retrieved_context = result.get("retrieved_context", "\n".join(chunk.get("text", "") for chunk in result.get("retrieved_chunks", [])))
    expected_keywords = expected_row.get("expected_keywords", "")
    expected_source = expected_row.get("expected_source", None)
    question_type = expected_row.get("question_type", "normal")
    is_unanswerable = str(question_type).lower().strip() == "unanswerable"
    return {
        "question": expected_row.get("question", result.get("question", "")),
        "question_type": question_type,
        "answer": answer,
        "expected_answer": expected_row.get("expected_answer", ""),
        "expected_source": expected_source,
        "retrieved_sources": ";".join(retrieved_sources),
        "source_match": source_match(retrieved_sources, expected_source),
        "keyword_recall": keyword_recall(answer, expected_keywords),
        "context_keyword_recall": None if is_unanswerable else context_keyword_recall(retrieved_context, expected_keywords),
        "answer_groundedness": answer_groundedness(answer, retrieved_context, question_type),
        "hallucination_risk": hallucination_risk(answer, retrieved_context, question_type),
        "source_citation": answer_has_source_citation(answer),
        "matched_keywords": ";".join(matched_keywords(answer, expected_keywords)),
        "missing_keywords": ";".join(missing_keywords(answer, expected_keywords)),
        "unanswerable_safe": answerability_check(answer, question_type),
    }


def evaluate_dataset(rag_pipeline, evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Run the RAG pipeline on all evaluation questions and calculate metrics."""
    required_columns = ["question", "expected_answer", "expected_source", "expected_keywords", "question_type"]
    missing = [col for col in required_columns if col not in evaluation_df.columns]
    if missing:
        raise ValueError(f"Evaluation dataset is missing columns: {missing}")
    rows = []
    for _, expected_row in evaluation_df.iterrows():
        result = rag_pipeline.ask(expected_row["question"])
        rows.append(evaluate_single_case(result, expected_row))
    return pd.DataFrame(rows)


def add_pass_fail_flags(
    evaluation_results: pd.DataFrame,
    keyword_recall_threshold: float = 0.5,
    context_recall_threshold: float = 0.5,
    groundedness_threshold: float = 0.8,
) -> pd.DataFrame:
    """Add pass/fail flags for retrieval, answer quality, grounding, and safety."""
    df = evaluation_results.copy()
    df["source_pass"] = df["source_match"].apply(lambda v: None if pd.isna(v) else int(v == 1))
    df["keyword_pass"] = df["keyword_recall"].apply(lambda v: None if pd.isna(v) else int(v >= keyword_recall_threshold))
    if "context_keyword_recall" in df.columns:
        df["context_pass"] = df["context_keyword_recall"].apply(lambda v: None if pd.isna(v) else int(v >= context_recall_threshold))
    else:
        df["context_pass"] = None
    if "answer_groundedness" in df.columns:
        df["groundedness_pass"] = df["answer_groundedness"].apply(lambda v: None if pd.isna(v) else int(v >= groundedness_threshold))
    else:
        df["groundedness_pass"] = None
    df["unanswerable_pass"] = df["unanswerable_safe"].apply(lambda v: None if pd.isna(v) else int(v == 1))

    def overall(row):
        checks = []
        for col in ["source_pass", "keyword_pass", "context_pass", "groundedness_pass", "unanswerable_pass"]:
            val = row[col]
            if not pd.isna(val):
                checks.append(int(val))
        if not checks:
            return None
        return int(all(v == 1 for v in checks))

    df["overall_pass"] = df.apply(overall, axis=1)
    return df


def summarize_results(evaluation_results: pd.DataFrame) -> pd.DataFrame:
    """Generate summary metrics for the evaluation run."""
    df = evaluation_results.copy()
    if "overall_pass" not in df.columns:
        df = add_pass_fail_flags(df)
    summary = {
        "total_questions": len(df),
        "answerable_questions": int((df["question_type"] != "unanswerable").sum()),
        "unanswerable_questions": int((df["question_type"] == "unanswerable").sum()),
        "avg_source_match": df["source_match"].mean(skipna=True),
        "avg_keyword_recall": df["keyword_recall"].mean(skipna=True),
        "avg_context_keyword_recall": df.get("context_keyword_recall", pd.Series(dtype=float)).mean(skipna=True),
        "avg_answer_groundedness": df.get("answer_groundedness", pd.Series(dtype=float)).mean(skipna=True),
        "avg_hallucination_risk": df.get("hallucination_risk", pd.Series(dtype=float)).mean(skipna=True),
        "avg_unanswerable_safe": df["unanswerable_safe"].mean(skipna=True),
        "overall_pass_rate": df["overall_pass"].mean(skipna=True),
    }
    return pd.DataFrame([summary])


def classify_failure_type(row: pd.Series) -> str:
    """Classify why a case failed."""
    if not pd.isna(row.get("source_pass")) and row.get("source_pass") == 0:
        return "Retrieval failure: expected source not retrieved"
    if not pd.isna(row.get("context_pass")) and row.get("context_pass") == 0:
        return "Retrieval failure: expected keywords missing from retrieved context"
    if not pd.isna(row.get("keyword_pass")) and row.get("keyword_pass") == 0:
        return "Answer failure: expected keywords missing"
    if not pd.isna(row.get("groundedness_pass")) and row.get("groundedness_pass") == 0:
        return "Grounding failure: answer not sufficiently supported by retrieved context"
    if not pd.isna(row.get("unanswerable_pass")) and row.get("unanswerable_pass") == 0:
        return "Safety failure: unanswerable question not handled safely"
    if row.get("overall_pass") == 0:
        return "Other failure"
    return "Passed"


def identify_failed_cases(evaluation_results: pd.DataFrame) -> pd.DataFrame:
    """Return failed cases with failure type labels."""
    df = evaluation_results.copy()
    if "overall_pass" not in df.columns:
        df = add_pass_fail_flags(df)
    df["failure_type"] = df.apply(classify_failure_type, axis=1)
    return df[df["overall_pass"] == 0].copy()


def save_evaluation_outputs(evaluation_results: pd.DataFrame, output_dir: str = "results") -> Dict[str, str]:
    """Save evaluation results, failed cases, and summary report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluation_results = add_pass_fail_flags(evaluation_results)
    failed_cases = identify_failed_cases(evaluation_results)
    summary = summarize_results(evaluation_results)
    evaluation_file = output_path / "evaluation_results.csv"
    failed_file = output_path / "failed_cases.csv"
    summary_file = output_path / "summary_report.csv"
    evaluation_results.to_csv(evaluation_file, index=False)
    failed_cases.to_csv(failed_file, index=False)
    summary.to_csv(summary_file, index=False)
    return {
        "evaluation_results": str(evaluation_file),
        "failed_cases": str(failed_file),
        "summary_report": str(summary_file),
    }
