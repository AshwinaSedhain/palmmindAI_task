"""
Plain dataclasses that represent rows from the SQLite tables.

We use simple dataclasses rather than an ORM to keep things easy to
understand — you can see exactly what goes into and comes out of the DB.
"""

from dataclasses import dataclass


@dataclass
class DocumentRecord:
    id: str
    filename: str
    strategy: str
    chunk_count: int
    created_at: str


@dataclass
class BookingRecord:
    id: str
    session_id: str
    full_name: str
    email: str
    preferred_date: str
    preferred_time: str
    created_at: str
