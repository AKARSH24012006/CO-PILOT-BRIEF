# CopilotBrief — Streaming Live RAG (Samsung PRISM GenAI Hackathon 3.0, Theme 4)
#
# Reproducibility (Gate G1): `docker compose up` launches the live demo
# server on a clean machine with no manual steps. The dense sentence
# embedding model is pre-warmed into the image below (network needed only
# at BUILD time); if that download fails — e.g. building on an offline
# network — the build still succeeds and the system falls back to a TF-IDF
# dense-analogue signal at runtime (see copilotbrief/retrieval.py), so the
# container always starts and always answers, either way.

FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scikit-learn/numpy wheels are prebuilt for slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm the sentence-transformers model cache at BUILD time so runtime
# start is instant and needs no network. Best-effort: `|| true` means an
# offline build still succeeds (see retrieval.py's TF-IDF fallback path).
RUN COPILOTBRIEF_ALLOW_MODEL_DOWNLOAD=1 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || true

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Default: launch the live streaming demo server. Override the command to
# run the CLI demo or benchmark instead, e.g.:
#   docker run <image> python run_demo.py
#   docker run <image> python benchmark/evaluate.py
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
