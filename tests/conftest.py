"""
pytest configuration.

Registers the 'integration' marker so tests can be filtered with:
    pytest -m "not integration"   # unit tests only
    pytest -m integration         # integration tests only
    pytest                        # all tests

Adds pytest-asyncio in auto mode so async test functions work without
decorating each one with @pytest.mark.asyncio.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live external services "
        "(Qdrant, Redis, SQLite, LLM API)",
    )
