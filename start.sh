#!/usr/bin/env bash
set -e

echo "🚀 Starting Research Mind AI Backend (FastAPI on :8000)..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

echo "✨ Starting Research Mind AI Frontend (Streamlit on :8501)..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &

# Wait for all background processes
wait -n
