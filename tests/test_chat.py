"""
Tests for the conversational RAG endpoint.

Unit tests (no external services):
  - test_is_booking_intent_*
  - test_booking_validation_*

Integration tests (require Redis + Qdrant + LLM API key):
  - test_chat_endpoint_*

Run unit tests only:
    pytest tests/test_chat.py -m "not integration"
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.booking import (
    is_booking_intent,
    validate_email,
    validate_date,
    validate_time,
    validate_name,
)


# ---------------------------------------------------------------------------
# Unit tests — booking intent detection
# ---------------------------------------------------------------------------

def test_is_booking_intent_book():
    assert is_booking_intent("I want to book an interview") is True


def test_is_booking_intent_schedule():
    assert is_booking_intent("Can we schedule a meeting?") is True


def test_is_booking_intent_unrelated():
    assert is_booking_intent("What is the capital of France?") is False


def test_is_booking_intent_case_insensitive():
    assert is_booking_intent("I need to BOOK an appointment") is True


# ---------------------------------------------------------------------------
# Unit tests — booking field validation
# ---------------------------------------------------------------------------

def test_validate_email_valid():
    assert validate_email("user@example.com") is True


def test_validate_email_invalid():
    assert validate_email("not-an-email") is False


def test_validate_email_none():
    assert validate_email(None) is False


def test_validate_date_future():
    assert validate_date("2099-12-31") is True


def test_validate_date_past():
    assert validate_date("2020-01-01") is False


def test_validate_date_invalid_format():
    assert validate_date("31/12/2099") is False


def test_validate_date_none():
    assert validate_date(None) is False


def test_validate_time_valid():
    assert validate_time("14:30") is True


def test_validate_time_midnight():
    assert validate_time("00:00") is True


def test_validate_time_invalid():
    assert validate_time("25:00") is False


def test_validate_time_none():
    assert validate_time(None) is False


def test_validate_name_two_words():
    assert validate_name("John Smith") is True


def test_validate_name_one_word():
    assert validate_name("John") is False


def test_validate_name_none():
    assert validate_name(None) is False


# ---------------------------------------------------------------------------
# Integration tests — require Redis + Qdrant + LLM
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_chat_endpoint_basic():
    """Send a message and verify the response structure."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "test-session-001", "message": "Hello, what can you help me with?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "session_id" in body
    assert body["session_id"] == "test-session-001"


@pytest.mark.integration
def test_chat_endpoint_booking_intent():
    """A booking-related message should trigger the booking flow."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "booking-test-001", "message": "I would like to book an interview"},
    )
    assert response.status_code == 200
    body = response.json()
    # Should ask for name (first missing field)
    assert "name" in body["answer"].lower()


@pytest.mark.integration
def test_chat_endpoint_missing_session_id():
    """Missing session_id should return 422."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 422


@pytest.mark.integration
def test_chat_multiturn_history():
    """Second message in same session should have access to history."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    session_id = "multiturn-test-001"

    client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "My name is Alice."},
    )
    response = client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "What is my name?"},
    )
    assert response.status_code == 200
    # The LLM should recall "Alice" from history
    body = response.json()
    assert "answer" in body
