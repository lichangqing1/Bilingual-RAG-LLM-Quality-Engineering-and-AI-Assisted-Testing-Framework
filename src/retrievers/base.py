from __future__ import annotations

from typing import Dict, List, Protocol


class BaseRetriever(Protocol):
    """Shared interface used by the RAG pipeline."""

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        """Build an index from text chunks."""

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        """Return top-k chunks for a query."""
