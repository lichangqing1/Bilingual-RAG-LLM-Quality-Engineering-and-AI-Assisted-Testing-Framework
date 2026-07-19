FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Optional FAISS/Sentence-Transformers image:
# COPY requirements-vector.txt ./
# RUN pip install --no-cache-dir -r requirements-vector.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
