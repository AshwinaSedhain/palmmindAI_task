"""
Tests for interview booking logic.

Unit tests (no external services):
  - Full booking state machine simulation using mocked Redis and SQLite
  - Duplicate booking prevention
  - Validation of all fields

Integration tests (require Redis + SQLite + LLM API):
  - Multi-turn booking conversation via the /chat endpoint
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.schemas.booking import BookingState
from app.services.booking import (
    validate_email,
    validate_date,
    validate_time,
    validate_name,
    is_booking_intent,
)


# ---------------------------------------------------------------------------
# Unit tests — state completeness checks
# ---------------------------------------------------------------------------

def test_complete_booking_state_all_valid():
    """A state with all valid fields should pass all validations."""
    state = BookingState(
        full_name="Jane Doe",
        email="jane@example.com",
        preferred_date="2099-06-15",
        preferred_time="10:00",
    )
    assert validate_name(state.full_name)
    assert validate_email(state.email)
    assert validate_date(state.preferred_date)
    assert validate_time(state.preferred_time)


def test_incomplete_booking_state_missing_email():
    state = BookingState(
        full_name="Jane Doe",
        email=None,
        preferred_date="2099-06-15",
        preferred_time="10:00",
    )
    assert validate_name(state.full_name)
    assert not validate_email(state.email)


def test_incomplete_booking_state_missing_date():
    state = BookingState(
        full_name="Jane Doe",
        email="jane@example.com",
        preferred_date=None,
        preferred_time="10:00",
    )
    assert not validate_date(state.preferred_date)


def test_booking_state_serialisation():
    """BookingState should serialise and deserialise correctly."""
    state = BookingState(
        full_name="John Smith",
        email="john@test.com",
        preferred_date="2099-01-20",
        preferred_time="09:30",
    )
    json_str = state.model_dump_json()
    restored = BookingState.model_validate_json(json_str)
    assert restored.full_name == state.full_name
    assert restored.email == state.email
    assert restored.preferred_date == state.preferred_date
    assert restored.preferred_time == state.preferred_time


# ---------------------------------------------------------------------------
# Unit tests — handle_booking (mocked dependencies)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_booking_asks_for_name_first():
    """When no booking state exists, handler should ask for name."""
    with (
        patch("app.services.booking.get_booking_state", return_value=BookingState()),
        patch("app.services.booking.extract_booking_info", return_value={
            "full_name": None, "email": None, "preferred_date": None, "preferred_time": None
        }),
        patch("app.services.booking.save_booking_state"),
    ):
        from app.services.booking import handle_booking
        response = await handle_booking("session-1", "I want to book an interview")
        assert "name" in response.lower()


@pytest.mark.asyncio
async def test_handle_booking_asks_for_email_after_name():
    """Once name is set, handler should ask for email."""
    state_with_name = BookingState(full_name="Alice Brown")
    with (
        patch("app.services.booking.get_booking_state", return_value=state_with_name),
        patch("app.services.booking.extract_booking_info", return_value={
            "full_name": None, "email": None, "preferred_date": None, "preferred_time": None
        }),
        patch("app.services.booking.save_booking_state"),
    ):
        from app.services.booking import handle_booking
        response = await handle_booking("session-1", "My name is Alice Brown")
        assert "email" in response.lower()


@pytest.mark.asyncio
async def test_handle_booking_asks_for_date_after_email():
    """Once name + email are set, handler should ask for date."""
    state = BookingState(full_name="Alice Brown", email="alice@example.com")
    with (
        patch("app.services.booking.get_booking_state", return_value=state),
        patch("app.services.booking.extract_booking_info", return_value={
            "full_name": None, "email": None, "preferred_date": None, "preferred_time": None
        }),
        patch("app.services.booking.save_booking_state"),
    ):
        from app.services.booking import handle_booking
        response = await handle_booking("session-1", "alice@example.com")
        assert "date" in response.lower()


@pytest.mark.asyncio
async def test_handle_booking_saves_when_complete():
    """When all fields are valid, booking should be saved and confirmed."""
    complete_state = BookingState(
        full_name="Alice Brown",
        email="alice@example.com",
        preferred_date="2099-06-20",
        preferred_time="14:00",
    )
    with (
        patch("app.services.booking.get_booking_state", return_value=complete_state),
        patch("app.services.booking.extract_booking_info", return_value={
            "full_name": None, "email": None,
            "preferred_date": None, "preferred_time": None,
        }),
        patch("app.services.booking.save_booking_state"),
        patch("app.services.booking.booking_exists", new_callable=AsyncMock, return_value=False),
        patch("app.services.booking.save_booking", new_callable=AsyncMock) as mock_save,
        patch("app.services.booking.clear_booking_state"),
    ):
        from app.schemas.booking import BookingConfirmation
        mock_save.return_value = BookingConfirmation(
            booking_id="test-uuid-123",
            full_name="Alice Brown",
            email="alice@example.com",
            preferred_date="2099-06-20",
            preferred_time="14:00",
        )
        from app.services.booking import handle_booking
        response = await handle_booking("session-1", "14:00")
        assert "booked successfully" in response.lower()
        assert "test-uuid-123" in response


@pytest.mark.asyncio
async def test_handle_booking_prevents_duplicate():
    """If an identical booking exists, it should inform the user without saving again."""
    complete_state = BookingState(
        full_name="Alice Brown",
        email="alice@example.com",
        preferred_date="2099-06-20",
        preferred_time="14:00",
    )
    with (
        patch("app.services.booking.get_booking_state", return_value=complete_state),
        patch("app.services.booking.extract_booking_info", return_value={
            "full_name": None, "email": None,
            "preferred_date": None, "preferred_time": None,
        }),
        patch("app.services.booking.save_booking_state"),
        patch("app.services.booking.booking_exists", new_callable=AsyncMock, return_value=True),
        patch("app.services.booking.save_booking", new_callable=AsyncMock) as mock_save,
        patch("app.services.booking.clear_booking_state"),
    ):
        from app.services.booking import handle_booking
        response = await handle_booking("session-1", "14:00")
        # Should NOT have called save_booking
        mock_save.assert_not_called()
        # Should mention the duplicate
        assert "already booked" in response.lower()


# ---------------------------------------------------------------------------
# Integration tests — require Redis + SQLite + LLM
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_booking_multiturn_via_chat():
    """
    Simulate a multi-turn booking conversation through the /chat endpoint.
    Requires: Redis running, SQLite writable, OpenAI API key.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    session_id = "booking-integration-001"

    # Turn 1: express intent
    r1 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "I would like to book an interview please",
    })
    assert r1.status_code == 200
    assert "name" in r1.json()["answer"].lower()

    # Turn 2: provide name
    r2 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "My name is John Smith",
    })
    assert r2.status_code == 200
    assert "email" in r2.json()["answer"].lower()

    # Turn 3: provide email
    r3 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "My email is john.smith@example.com",
    })
    assert r3.status_code == 200
    assert "date" in r3.json()["answer"].lower()

    # Turn 4: provide date
    r4 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "I prefer 2099-08-15",
    })
    assert r4.status_code == 200
    assert "time" in r4.json()["answer"].lower()

    # Turn 5: provide time — should confirm booking
    r5 = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "10:00 would be great",
    })
    assert r5.status_code == 200
    assert "booked successfully" in r5.json()["answer"].lower()
