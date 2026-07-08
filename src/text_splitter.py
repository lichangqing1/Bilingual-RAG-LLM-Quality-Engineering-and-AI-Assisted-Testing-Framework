from typing import Dict, List


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Parameters
    ----------
    text:
        Input document text.
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters shared between consecutive chunks.

    Returns
    -------
    List[str]
        Text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, str]]:
    """
    Convert loaded documents into chunks with metadata.
    """
    all_chunks = []

    for doc in documents:
        chunks = split_text(
            text=doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['source']}_chunk_{i}",
                "source": doc["source"],
                "text": chunk_text,
                "chunk_index": i,
                "num_characters": len(chunk_text),
                "num_words": len(chunk_text.split())
            })

    return all_chunks


def get_chunk_statistics(chunks: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Return basic chunk statistics.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_characters": 0,
            "avg_words": 0,
            "min_characters": 0,
            "max_characters": 0
        }

    character_lengths = [chunk["num_characters"] for chunk in chunks]
    word_lengths = [chunk["num_words"] for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_characters": sum(character_lengths) / len(character_lengths),
        "avg_words": sum(word_lengths) / len(word_lengths),
        "min_characters": min(character_lengths),
        "max_characters": max(character_lengths)
    }
