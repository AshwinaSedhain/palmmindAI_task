#!/bin/bash
# Start script - runs required services and the backend

echo "Starting RAG Backend..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.example to .env and add your GROQ_API_KEY"
    exit 1
fi

# Check if Qdrant is running
if ! curl -s http://localhost:6333 > /dev/null 2>&1; then
    echo "Error: Qdrant is not running on port 6333"
    echo "Install and start Qdrant:"
    echo "  pip install qdrant-client[fastembed]"
    echo "  qdrant-server"
    echo "Or use binary: https://qdrant.tech/documentation/guides/installation/"
    exit 1
fi

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Error: Redis is not running"
    echo "Start Redis with: redis-server"
    exit 1
fi

echo "Qdrant: OK"
echo "Redis: OK"
echo "Starting FastAPI server on http://localhost:8000"
echo ""

# Kill any process using port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
