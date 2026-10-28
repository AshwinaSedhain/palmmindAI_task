"""
Pydantic schemas for the document ingestion endpoint.

IngestResponse is what the client receives after a successful upload.
"""

from pydantic import BaseModel
from typing import Literal


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: Literal["fixed", "sentence"]
    chunk_count: int
    message: str
