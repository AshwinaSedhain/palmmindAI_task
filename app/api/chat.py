import asyncio
from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.retrieval import retrieve_relevant_chunks
from app.services.conversation import get_chat_history, save_chat_message
from app.services.llm import generate_rag_answer
from app.services.booking import is_booking_intent, handle_booking

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message
    
    # check if this is a booking request
    if is_booking_intent(message):
        try:
            response = await handle_booking(session_id, message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Booking error: {e}")
        
        save_chat_message(session_id, "user", message)
        save_chat_message(session_id, "assistant", response)
        
        return ChatResponse(
            session_id=session_id,
            answer=response,
            sources_used=0,
        )
    
    # normal RAG flow
    try:
        chunks = retrieve_relevant_chunks(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")
    
    history = get_chat_history(session_id)
    context_texts = [c["text"] for c in chunks]
    
    try:
        # run in thread pool since openai client is sync
        answer = await asyncio.to_thread(
            generate_rag_answer,
            question=message,
            context_chunks=context_texts,
            chat_history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    
    save_chat_message(session_id, "user", message)
    save_chat_message(session_id, "assistant", answer)
    
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources_used=len(chunks),
    )
