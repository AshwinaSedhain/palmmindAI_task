from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import init_db
from app.services.vector_store import ensure_collection_exists
from app.api import ingestion, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await init_db()
    ensure_collection_exists()
    yield
    # shutdown (nothing needed)


app = FastAPI(
    title="RAG Backend API",
    description="Document ingestion + conversational RAG + interview booking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    ingestion.router,
    prefix="/api/v1/documents",
    tags=["Ingestion"],
)

app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"],
)


@app.get("/")
async def root():
    return {"status": "ok"}
