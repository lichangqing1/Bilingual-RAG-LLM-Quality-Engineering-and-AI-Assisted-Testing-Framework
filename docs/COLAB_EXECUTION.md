# Colab Execution Script

This guide provides copy-paste cells for running `rag-evaluation-framework` in Google Colab.

Use the lightweight default path first. It runs the local semantic backend, hybrid retrieval, pytest, full evaluation, regression gates, and security evaluation without installing FAISS or downloading Sentence-Transformers models.

## Option A: Upload Project Zip

Use this when you have a local zip file such as `rag-evaluation-framework-clean-retrievers.zip`.

```python
from google.colab import files

uploaded = files.upload()
zip_name = next(iter(uploaded))
print("Uploaded:", zip_name)
```

```python
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_DIR = Path("/content/rag-evaluation-framework")

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)

with zipfile.ZipFile(zip_name, "r") as archive:
    archive.extractall("/content")

if not PROJECT_DIR.exists():
    candidates = [p for p in Path("/content").iterdir() if p.is_dir() and "rag" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError("Could not find extracted project folder under /content.")
    PROJECT_DIR = candidates[0]

os.chdir(PROJECT_DIR)
print("Project directory:", PROJECT_DIR)
print("Files:", sorted(p.name for p in PROJECT_DIR.iterdir())[:20])
```

## Option B: Clone From GitHub

Use this after pushing the project to GitHub.

```python
import os
import shutil
from pathlib import Path

REPO_URL = "https://github.com/YOUR_USERNAME/rag-evaluation-framework.git"
PROJECT_DIR = Path("/content/rag-evaluation-framework")

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)

!git clone {REPO_URL} {PROJECT_DIR}
os.chdir(PROJECT_DIR)
print("Project directory:", PROJECT_DIR)
```

## Install Lightweight Dependencies

```python
!python --version
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

## Optional: Install Vector Dependencies

Only run this cell if you want `semantic_backend=faiss`.

```python
!pip install -r requirements-vector.txt
```

## Run Tests

```python
!pytest -q
```

Expected result:

```text
43 passed, 1 skipped
```

## Run Full RAG Evaluation

```python
!python scripts/run_evaluation.py --retrieval-mode hybrid --semantic-backend local
```

Show the summary:

```python
import pandas as pd

summary = pd.read_csv("results/summary_report.csv")
summary
```

## Run Security Evaluation

```python
!python scripts/run_security_evaluation.py --retrieval-mode hybrid --semantic-backend local
```

Show the security summary:

```python
import pandas as pd

security_summary = pd.read_csv("results/security_summary.csv")
security_summary
```

## Run Regression Gates

Run this after the full evaluation has generated `results/summary_report.csv`.

```python
!python scripts/run_regression_checks.py
```

## Try the RAG Pipeline Directly

```python
from src.document_loader import load_markdown_documents, validate_documents
from src.text_splitter import create_chunks
from src.retrieval import build_vector_store
from src.rag_pipeline import SimpleRAGPipeline

docs = load_markdown_documents("data/documents")
validate_documents(docs)
chunks = create_chunks(docs, chunk_size=500, overlap=100)

vector_store = build_vector_store(retrieval_mode="hybrid", semantic_backend="local")
vector_store.build_index(chunks)

rag = SimpleRAGPipeline(vector_store, top_k=3)
response = rag.ask("Can I return a product after 90 days?")

print(response["answer"])
print(response["sources"])
```

## Optional: Run FAISS Semantic Retrieval

Run the vector dependency install cell first.

```python
!python scripts/run_evaluation.py --retrieval-mode semantic --semantic-backend faiss
```

## Optional: Run Streamlit in Colab

Colab does not expose local ports directly, so use `pyngrok`.

```python
!pip install pyngrok
```

```python
from pyngrok import ngrok
import subprocess
import time

process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

time.sleep(5)
public_url = ngrok.connect(8501)
print("Streamlit URL:", public_url)
```

Stop Streamlit when finished:

```python
process.terminate()
ngrok.kill()
```

## Download Results

```python
from google.colab import files

files.download("results/summary_report.csv")
files.download("results/security_summary.csv")
files.download("results/evaluation_report.md")
```
