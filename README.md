# RAG Backend - Interview Assignment

Backend API for document ingestion, conversational RAG, and interview booking.

## Prerequisites

```bash
# Install Redis
sudo pacman -S redis  # Arch Linux
# or: sudo apt install redis  # Ubuntu

# Install Qdrant
# Download from: https://github.com/qdrant/qdrant/releases
# Or use Python: pip install qdrant-client[fastembed]
```

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Add Groq API key to .env
nano .env
```

## Run

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Qdrant (download binary from qdrant.tech)
./qdrant

# Terminal 3: Start backend
./run.sh
```

Server: http://localhost:8000  
API Docs: http://localhost:8000/docs

## Test

```bash
# Unit tests
pytest -m "not integration"

# All tests (requires services running)
pytest
```

## Endpoints

**POST /api/v1/documents/ingest**
- Upload PDF/TXT files
- Chunking strategies: fixed, sentence

**POST /api/v1/chat**  
- Conversational RAG
- Interview booking

## Stack

FastAPI, PyMuPDF, sentence-transformers, Qdrant, Redis, SQLite, Groq
