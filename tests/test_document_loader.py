from src.document_loader import load_markdown_documents, validate_documents, get_document_summary


def test_load_markdown_documents():
    docs = load_markdown_documents("data/documents")
    assert len(docs) > 0
    assert "source" in docs[0]
    assert "text" in docs[0]
    assert docs[0]["source"].endswith(".md")
    assert len(docs[0]["text"]) > 0


def test_validate_documents():
    docs = load_markdown_documents("data/documents")
    validate_documents(docs)


def test_get_document_summary():
    docs = load_markdown_documents("data/documents")
    summary = get_document_summary(docs)
    assert len(summary) == len(docs)
    assert "source" in summary[0]
    assert "num_words" in summary[0]
