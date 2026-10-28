import os
import aiosqlite
from app.core.config import settings


def get_db_path():
    path = settings.sqlite_db_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


async def get_connection():
    """Get a new database connection each time"""
    conn = await aiosqlite.connect(get_db_path())
    conn.row_factory = aiosqlite.Row
    return conn


async def init_db():
    """Setup tables on startup"""
    conn = await aiosqlite.connect(get_db_path())
    conn.row_factory = aiosqlite.Row
    
    try:
        # documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                strategy TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # bookings table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL,
                preferred_date TEXT NOT NULL,
                preferred_time TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await conn.commit()
    finally:
        await conn.close()
