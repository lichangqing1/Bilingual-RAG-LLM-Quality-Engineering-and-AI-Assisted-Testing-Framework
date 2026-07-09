from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfVectorStore:
    """
    Lightweight vector store for local demos and regression evaluation.

    This avoids native FAISS/PyTorch dependencies while preserving the same
    search interface used by SimpleRAGPipeline.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
        )
        self.chunks = None
        self.matrix = None

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        if not chunks:
            raise ValueError("No chunks provided. Cannot build vector index.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        if self.matrix is None or self.chunks is None:
            raise ValueError("Index has not been built yet.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.chunks))
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)].copy()
            chunk["similarity_score"] = float(scores[int(idx)])
            results.append(chunk)

        return results
