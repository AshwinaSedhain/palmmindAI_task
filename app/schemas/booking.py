"""
Pydantic schemas for interview booking.

BookingState holds the information we are collecting across turns.
BookingConfirmation is what we store and return when a booking is complete.
"""

from pydantic import BaseModel, EmailStr
from typing import Optional


class BookingState(BaseModel):
    """Partial booking info collected so far in a conversation session."""
    full_name: Optional[str] = None
    email: Optional[str] = None
    preferred_date: Optional[str] = None   # stored as ISO date string YYYY-MM-DD
    preferred_time: Optional[str] = None   # stored as HH:MM (24-hour)


class BookingConfirmation(BaseModel):
    """A fully validated, saved booking."""
    booking_id: str
    full_name: str
    email: str
    preferred_date: str
    preferred_time: str
