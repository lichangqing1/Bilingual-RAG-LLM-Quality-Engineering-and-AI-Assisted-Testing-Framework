import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    """
    FAISS-based vector store for semantic retrieval.

    This implementation uses cosine similarity by normalizing embeddings and
    searching with inner product.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = None
        self.embeddings = None

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        """
        Build FAISS index from text chunks.
        """
        if not chunks:
            raise ValueError("No chunks provided. Cannot build vector index.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        self.embeddings = embeddings

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Inner Product
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        """
        Search top-k most relevant chunks for a query.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.chunks))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results

    def save(self, output_dir: str) -> None:
        """
        Save FAISS index and chunk metadata.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output_path / "faiss.index"))

        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "chunks": self.chunks
        }

        with open(output_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, input_dir: str):
        """
        Load FAISS index and chunk metadata.
        """
        input_path = Path(input_dir)

        with open(input_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        store = cls(embedding_model_name=metadata["embedding_model_name"])
        store.index = faiss.read_index(str(input_path / "faiss.index"))
        store.chunks = metadata["chunks"]

        return store
