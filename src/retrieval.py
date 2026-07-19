from __future__ import annotations

from typing import Literal

from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.lexical_retriever import LexicalRetriever
from src.retrievers.semantic_retriever import LocalSemanticRetriever, SemanticFaissRetriever


RetrievalMode = Literal["lexical", "keyword", "semantic", "hybrid"]
SemanticBackend = Literal["local", "faiss"]


def build_vector_store(
    retrieval_mode: RetrievalMode = "hybrid",
    semantic_backend: SemanticBackend = "local",
):
    """
    Build a retriever for the requested evaluation mode.

    Modes:
    - lexical/keyword: BM25/TF-IDF only
    - semantic: Sentence-Transformers + FAISS when backend is faiss; local dense fallback otherwise
    - hybrid: lexical retrieval + semantic retrieval + score fusion
    """
    if retrieval_mode in {"lexical", "keyword"}:
        return LexicalRetriever()

    if retrieval_mode == "semantic":
        if semantic_backend == "faiss":
            return SemanticFaissRetriever()
        return LocalSemanticRetriever()

    if retrieval_mode == "hybrid":
        return HybridRetriever(semantic_backend=semantic_backend)

    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
