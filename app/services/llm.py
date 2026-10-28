import json
import re
from openai import OpenAI
from app.core.config import settings


def get_client():
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.llm_base_url,
    )


def generate_rag_answer(question: str, context_chunks: list[str], chat_history: list[dict]):
    client = get_client()
    
    system_msg = (
        "You are a helpful assistant. Answer based on the provided context. "
        "If you don't have enough information, say so."
    )
    
    # build context
    if context_chunks:
        ctx = "\n\n---\n\n".join(
            f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(context_chunks)
        )
        context_block = f"Context:\n\n{ctx}"
    else:
        context_block = "No relevant context found."
    
    messages = [{"role": "system", "content": system_msg}]
    
    # add recent history
    recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
    messages.extend(recent)
    
    # current question with context
    user_msg = f"{context_block}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_msg})
    
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
    )
    
    return response.choices[0].message.content.strip()


def extract_booking_info(message: str):
    """Use LLM to pull out booking fields from natural language"""
    client = get_client()
    
    prompt = """Extract booking info from the message. Return JSON only:
{
  "full_name": "string or null",
  "email": "string or null", 
  "preferred_date": "YYYY-MM-DD or null",
  "preferred_time": "HH:MM or null"
}

Rules:
- Only extract what's explicitly mentioned
- Convert dates to YYYY-MM-DD
- Convert times to 24-hour HH:MM
- Don't guess or invent values
"""
    
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    
    raw = response.choices[0].message.content.strip()
    
    # strip code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    
    try:
        data = json.loads(raw)
    except:
        # fallback: try to find json block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                data = {}
        else:
            data = {}
    
    return {
        "full_name": data.get("full_name"),
        "email": data.get("email"),
        "preferred_date": data.get("preferred_date"),
        "preferred_time": data.get("preferred_time"),
    }
