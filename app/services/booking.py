import re
import uuid
from datetime import datetime, date

from app.schemas.booking import BookingState, BookingConfirmation
from app.services.conversation import get_booking_state, save_booking_state, clear_booking_state
from app.services.llm import extract_booking_info
from app.db.database import get_connection


BOOKING_KEYWORDS = [
    "book", "schedule", "interview", "appointment", "reserve",
    "set up", "arrange", "meeting", "slot",
]


def is_booking_intent(msg: str):
    lower = msg.lower()
    return any(kw in lower for kw in BOOKING_KEYWORDS)


# validators
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def validate_email(email):
    return bool(email and EMAIL_RE.match(email))


def validate_date(date_str):
    if not date_str:
        return False
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
        return parsed >= date.today()
    except:
        return False


def validate_time(time_str):
    if not time_str:
        return False
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except:
        return False


def validate_name(name):
    if not name:
        return False
    return len(name.strip().split()) >= 2


async def booking_exists(email: str, pref_date: str, pref_time: str):
    async with await get_connection() as conn:
        cursor = await conn.execute(
            "SELECT 1 FROM bookings WHERE email=? AND preferred_date=? AND preferred_time=? LIMIT 1",
            (email, pref_date, pref_time),
        )
        row = await cursor.fetchone()
        return row is not None


async def save_booking(session_id: str, state: BookingState):
    booking_id = str(uuid.uuid4())
    created = datetime.utcnow().isoformat()
    
    async with await get_connection() as conn:
        await conn.execute(
            """INSERT INTO bookings 
               (id, session_id, full_name, email, preferred_date, preferred_time, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (booking_id, session_id, state.full_name, state.email, 
             state.preferred_date, state.preferred_time, created),
        )
        await conn.commit()
    
    return BookingConfirmation(
        booking_id=booking_id,
        full_name=state.full_name,
        email=state.email,
        preferred_date=state.preferred_date,
        preferred_time=state.preferred_time,
    )


async def handle_booking(session_id: str, message: str):
    # load current state
    state = get_booking_state(session_id)
    
    # extract new info
    extracted = extract_booking_info(message)
    
    # merge
    if extracted.get("full_name"):
        state.full_name = extracted["full_name"]
    if extracted.get("email"):
        state.email = extracted["email"]
    if extracted.get("preferred_date"):
        state.preferred_date = extracted["preferred_date"]
    if extracted.get("preferred_time"):
        state.preferred_time = extracted["preferred_time"]
    
    save_booking_state(session_id, state)
    
    # check what's missing
    if not validate_name(state.full_name):
        return "I can help you book an interview! What's your full name?"
    
    if not validate_email(state.email):
        return f"Thanks {state.full_name}. What's your email?"
    
    if not validate_date(state.preferred_date):
        return "What date works for you? (YYYY-MM-DD format)"
    
    if not validate_time(state.preferred_time):
        return "What time? (HH:MM format, like 14:00)"
    
    # all valid - check duplicates
    if await booking_exists(state.email, state.preferred_date, state.preferred_time):
        clear_booking_state(session_id)
        return (
            f"Looks like {state.email} already has an interview on "
            f"{state.preferred_date} at {state.preferred_time}. "
            f"Pick a different time?"
        )
    
    # save it
    confirmation = await save_booking(session_id, state)
    clear_booking_state(session_id)
    
    return (
        f"Interview booked!\n\n"
        f"ID: {confirmation.booking_id}\n"
        f"Name: {confirmation.full_name}\n"
        f"Email: {confirmation.email}\n"
        f"Date: {confirmation.preferred_date}\n"
        f"Time: {confirmation.preferred_time}\n\n"
        f"Good luck!"
    )
