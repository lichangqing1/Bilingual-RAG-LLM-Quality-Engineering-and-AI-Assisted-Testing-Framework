import pytest

from src.text_splitter import split_text, create_chunks, get_chunk_statistics


def test_split_text_returns_chunks():
    text = "This is a test document. " * 100
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_size_must_be_larger_than_overlap():
    with pytest.raises(ValueError):
        split_text("sample text", chunk_size=50, overlap=50)


def test_create_chunks():
    documents = [{"source": "sample.md", "text": "This is a sample document. " * 50}]
    chunks = create_chunks(documents, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert "chunk_id" in chunks[0]
    assert "source" in chunks[0]
    assert "text" in chunks[0]


def test_get_chunk_statistics():
    documents = [{"source": "sample.md", "text": "This is a sample document. " * 50}]
    chunks = create_chunks(documents, chunk_size=100, overlap=20)
    stats = get_chunk_statistics(chunks)
    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_characters"] > 0
