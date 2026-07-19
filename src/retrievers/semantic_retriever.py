from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.retrievers.lexical_retriever import LexicalRetriever


class LocalSemanticRetriever:
    """CI-friendly semantic baseline using TF-IDF + SVD embeddings."""

    def __init__(self, svd_components: int = 64):
        self.svd_components = svd_components
        self.lexical = LexicalRetriever()
        self.chunks = None
        self.svd = None
        self.embedding_matrix = None

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        self.lexical.build_index(chunks)
        self.chunks = chunks
        n_docs, n_features = self.lexical.tfidf_matrix.shape
        if min(n_docs, n_features) <= 2:
            self.svd = None
            self.embedding_matrix = self.lexical.tfidf_matrix
            return

        n_components = min(self.svd_components, n_docs - 1, n_features - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.embedding_matrix = self.svd.fit_transform(self.lexical.tfidf_matrix)

    def scores(self, query: str) -> np.ndarray:
        if self.chunks is None:
            raise ValueError("Index has not been built yet.")
        query_tfidf = self.lexical.vectorizer.transform([query])
        if self.svd is not None:
            query_embedding = self.svd.transform(query_tfidf)
        else:
            query_embedding = query_tfidf
        return cosine_similarity(query_embedding, self.embedding_matrix)[0]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        if self.chunks is None:
            raise ValueError("Index has not been built yet.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.chunks))
        raw_scores = self.scores(query)
        normalized = LexicalRetriever.normalize_scores(raw_scores)
        ranked_indices = normalized.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)].copy()
            chunk["semantic_score"] = float(normalized[int(idx)])
            chunk["similarity_score"] = float(normalized[int(idx)])
            chunk["retrieval_mode"] = "semantic"
            chunk["retrieval_method"] = "semantic_local_tfidf_svd"
            results.append(chunk)
        return results


class SemanticFaissRetriever:
    """Main semantic retriever: Sentence-Transformers embeddings + FAISS vector search."""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic FAISS retrieval requires sentence-transformers. "
                "Install optional vector dependencies with "
                "`pip install -r requirements-vector.txt`."
            ) from exc

        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = None
        self.embeddings = None

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        if not chunks:
            raise ValueError("No chunks provided. Cannot build semantic index.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        import faiss

        faiss.normalize_L2(embeddings)

        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        if self.index is None:
            raise ValueError("Index has not been built yet.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.chunks))
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        import faiss

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["semantic_score"] = float(score)
            chunk["similarity_score"] = float(score)
            chunk["retrieval_mode"] = "semantic"
            chunk["retrieval_method"] = "semantic_faiss_sentence_transformers"
            results.append(chunk)
        return results

    def save(self, output_dir: str) -> None:
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        import faiss

        faiss.write_index(self.index, str(output_path / "faiss.index"))
        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "chunks": self.chunks,
        }
        with open(output_path / "metadata.pkl", "wb") as file:
            pickle.dump(metadata, file)

    @classmethod
    def load(cls, input_dir: str):
        input_path = Path(input_dir)
        with open(input_path / "metadata.pkl", "rb") as file:
            metadata = pickle.load(file)
        store = cls(embedding_model_name=metadata["embedding_model_name"])
        import faiss

        store.index = faiss.read_index(str(input_path / "faiss.index"))
        store.chunks = metadata["chunks"]
        return store
