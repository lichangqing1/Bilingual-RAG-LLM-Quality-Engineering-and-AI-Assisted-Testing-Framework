from pathlib import Path
from typing import Dict, List, Sequence


SUPPORTED_EXTENSIONS = (".md", ".txt")


def load_documents(
    document_dir: str,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS
) -> List[Dict[str, str]]:
    """
    Load text-based documents from a folder.

    Parameters
    ----------
    document_dir:
        Folder that contains knowledge base documents.
    extensions:
        File extensions to load. Default supports Markdown and TXT.

    Returns
    -------
    List[dict]
        Each document is represented as:
        {
            "source": "return_policy.md",
            "source_path": ".../return_policy.md",
            "text": "document content"
        }
    """
    document_path = Path(document_dir)

    if not document_path.exists():
        raise FileNotFoundError(f"Document folder not found: {document_dir}")

    documents = []

    for file_path in sorted(document_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            text = file_path.read_text(encoding="utf-8").strip()

            if text:
                documents.append({
                    "source": file_path.name,
                    "source_path": str(file_path),
                    "text": text
                })

    return documents


def load_markdown_documents(document_dir: str) -> List[Dict[str, str]]:
    """
    Load only Markdown documents from a folder.
    """
    return load_documents(document_dir=document_dir, extensions=(".md",))


def validate_documents(documents: List[Dict[str, str]]) -> None:
    """
    Validate loaded documents before chunking or indexing.
    """
    if not documents:
        raise ValueError("No documents were loaded. Please check the document folder.")

    required_keys = {"source", "text"}

    for i, doc in enumerate(documents):
        missing_keys = required_keys - set(doc.keys())

        if missing_keys:
            raise ValueError(f"Document {i} is missing keys: {missing_keys}")

        if not doc["text"].strip():
            raise ValueError(f"Document {doc.get('source', i)} has empty text.")

    print(f"Document validation passed. Loaded {len(documents)} documents.")


def get_document_summary(documents: List[Dict[str, str]]):
    """
    Return summary information for loaded documents.
    """
    return [
        {
            "source": doc["source"],
            "num_characters": len(doc["text"]),
            "num_words": len(doc["text"].split())
        }
        for doc in documents
    ]
