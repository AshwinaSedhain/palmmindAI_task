"""
Pydantic schemas for the conversational RAG endpoint.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources_used: int  # number of retrieved chunks used as context
