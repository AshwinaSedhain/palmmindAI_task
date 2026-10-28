import json
import redis
from app.core.config import settings
from app.schemas.booking import BookingState


def get_redis_client():
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


# chat history
CHAT_PREFIX = "chat:"
MAX_HISTORY = 20


def get_chat_history(session_id: str):
    client = get_redis_client()
    key = CHAT_PREFIX + session_id
    data = client.get(key)
    if data is None:
        return []
    return json.loads(data)


def save_chat_message(session_id: str, role: str, content: str):
    client = get_redis_client()
    key = CHAT_PREFIX + session_id
    
    history = get_chat_history(session_id)
    history.append({"role": role, "content": content})
    
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    
    client.set(key, json.dumps(history))


# booking state (separate from chat)
BOOKING_PREFIX = "booking:"


def get_booking_state(session_id: str):
    client = get_redis_client()
    key = BOOKING_PREFIX + session_id
    data = client.get(key)
    if data is None:
        return BookingState()
    return BookingState(**json.loads(data))


def save_booking_state(session_id: str, state: BookingState):
    client = get_redis_client()
    key = BOOKING_PREFIX + session_id
    client.set(key, state.model_dump_json())


def clear_booking_state(session_id: str):
    client = get_redis_client()
    key = BOOKING_PREFIX + session_id
    client.delete(key)
