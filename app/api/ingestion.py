import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status

from app.core.config import settings
from app.schemas.ingestion import IngestResponse
from app.services.document_loader import extract_text, validate_file_extension
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import upsert_chunks
from app.db.database import get_connection

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: Annotated[UploadFile, File()],
    chunking_strategy: Annotated[Literal["fixed", "sentence"], Form()] = "fixed",
    chunk_size: Annotated[int, Form(ge=50, le=10000)] = settings.default_chunk_size,
    chunk_overlap: Annotated[int, Form(ge=0, le=500)] = settings.default_chunk_overlap,
):
    filename = file.filename or "unknown"
    
    # validate extension
    try:
        validate_file_extension(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # extract text
    try:
        text = await extract_text(file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # chunk
    try:
        chunks = chunk_text(text, strategy=chunking_strategy, 
                           chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks created")
    
    # embed
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    
    # store in vector db
    doc_id = str(uuid.uuid4())
    try:
        upsert_chunks(doc_id, chunks, embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store failed: {e}")
    
    # save metadata
    created = datetime.utcnow().isoformat()
    try:
        async with await get_connection() as conn:
            await conn.execute(
                "INSERT INTO documents (id, filename, strategy, chunk_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, chunking_strategy, len(chunks), created),
            )
            await conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    
    return IngestResponse(
        document_id=doc_id,
        filename=filename,
        chunking_strategy=chunking_strategy,
        chunk_count=len(chunks),
        message=f"Ingested {len(chunks)} chunks",
    )
