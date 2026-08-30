# Research Mind AI - Multi-Stage Production Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for PDF processing & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code
COPY . .

# Ensure storage directories and permissions
RUN mkdir -p /app/data/chromadb /app/data/uploads && \
    chmod +x /app/start.sh

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command starts both backend and frontend
CMD ["/app/start.sh"]
