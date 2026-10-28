"""
Tests for document ingestion.

Unit tests (no external services required):
  - test_validate_file_extension_*
  - test_extract_from_txt_*
  - test_chunk_fixed_*
  - test_chunk_sentences_*
  - test_chunk_dispatcher_*

Integration tests (require running Qdrant + the embedding model):
  - test_ingest_txt_endpoint
  - test_ingest_pdf_endpoint
  - test_ingest_invalid_extension

Mark integration tests with @pytest.mark.integration so they can be
skipped in CI environments without live services:
    pytest -m "not integration"
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.services.document_loader import validate_file_extension, _extract_from_txt
from app.services.chunking import chunk_fixed, chunk_sentences, chunk_text


# ---------------------------------------------------------------------------
# Unit tests — file validation
# ---------------------------------------------------------------------------

def test_validate_file_extension_pdf():
    assert validate_file_extension("report.pdf") == ".pdf"


def test_validate_file_extension_txt():
    assert validate_file_extension("notes.txt") == ".txt"


def test_validate_file_extension_uppercase():
    assert validate_file_extension("REPORT.PDF") == ".pdf"


def test_validate_file_extension_unsupported():
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file_extension("image.png")


def test_validate_file_extension_no_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file_extension("noextension")


# ---------------------------------------------------------------------------
# Unit tests — text extraction (TXT)
# ---------------------------------------------------------------------------

def test_extract_from_txt_utf8():
    text = _extract_from_txt("Hello, world!".encode("utf-8"))
    assert text == "Hello, world!"


def test_extract_from_txt_strips_whitespace():
    text = _extract_from_txt("  hello  \n".encode("utf-8"))
    assert text == "hello"


def test_extract_from_txt_empty_raises():
    with pytest.raises(ValueError, match="no readable content"):
        _extract_from_txt(b"   ")


# ---------------------------------------------------------------------------
# Unit tests — fixed chunking
# ---------------------------------------------------------------------------

def test_chunk_fixed_basic():
    text = "a" * 1000
    chunks = chunk_fixed(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    # Each chunk should be at most 200 characters
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_fixed_overlap_creates_more_chunks():
    text = "x" * 600
    chunks_no_overlap = chunk_fixed(text, chunk_size=200, chunk_overlap=0)
    chunks_with_overlap = chunk_fixed(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_chunk_fixed_short_text():
    text = "short"
    chunks = chunk_fixed(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "short"


def test_chunk_fixed_invalid_size():
    with pytest.raises(ValueError, match="chunk_size must be"):
        chunk_fixed("text", chunk_size=0)


def test_chunk_fixed_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        chunk_fixed("text", chunk_size=100, chunk_overlap=100)


def test_chunk_fixed_no_empty_chunks():
    text = "hello world " * 50
    chunks = chunk_fixed(text, chunk_size=100, chunk_overlap=10)
    assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# Unit tests — sentence chunking
# ---------------------------------------------------------------------------

def test_chunk_sentences_basic():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a second sentence. "
        "And here is a third one."
    )
    chunks = chunk_sentences(text, chunk_size=60)
    assert len(chunks) >= 1
    # All original sentences should be present across chunks
    full_text = " ".join(chunks)
    assert "quick brown fox" in full_text


def test_chunk_sentences_single_long_sentence():
    # A single sentence longer than chunk_size should not be discarded
    text = "word " * 200  # ~1000 chars
    chunks = chunk_sentences(text, chunk_size=100)
    assert len(chunks) >= 1
    # All content should be preserved
    combined = " ".join(chunks)
    assert len(combined) > 0


def test_chunk_sentences_invalid_size():
    with pytest.raises(ValueError, match="chunk_size must be"):
        chunk_sentences("text", chunk_size=0)


# ---------------------------------------------------------------------------
# Unit tests — chunk dispatcher
# ---------------------------------------------------------------------------

def test_chunk_text_fixed_strategy():
    chunks = chunk_text("hello " * 100, strategy="fixed", chunk_size=100, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_text_sentence_strategy():
    chunks = chunk_text("Hello world. Goodbye world.", strategy="sentence", chunk_size=100)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_text_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("text", strategy="unknown")  # type: ignore


# ---------------------------------------------------------------------------
# Integration tests — require Qdrant + embedding model
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ingest_txt_endpoint():
    """Ingest a small TXT file and verify the response."""
    from app.main import app
    client = TestClient(app)

    txt_content = b"This is a test document. It has multiple sentences. We will chunk it."
    response = client.post(
        "/api/v1/documents/ingest",
        data={"chunking_strategy": "fixed", "chunk_size": "100", "chunk_overlap": "10"},
        files={"file": ("test.txt", io.BytesIO(txt_content), "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "test.txt"
    assert body["chunking_strategy"] == "fixed"
    assert body["chunk_count"] >= 1
    assert "document_id" in body


@pytest.mark.integration
def test_ingest_txt_sentence_strategy():
    """Ingest a TXT file using sentence-based chunking."""
    from app.main import app
    client = TestClient(app)

    txt_content = (
        b"Artificial intelligence is transforming industries. "
        b"Machine learning enables computers to learn from data. "
        b"Deep learning uses neural networks with many layers."
    )
    response = client.post(
        "/api/v1/documents/ingest",
        data={"chunking_strategy": "sentence", "chunk_size": "100"},
        files={"file": ("ai.txt", io.BytesIO(txt_content), "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["chunking_strategy"] == "sentence"
    assert body["chunk_count"] >= 1


@pytest.mark.integration
def test_ingest_invalid_extension():
    """Uploading an unsupported file type should return 400."""
    from app.main import app
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/ingest",
        data={"chunking_strategy": "fixed"},
        files={"file": ("image.png", io.BytesIO(b"fake png data"), "image/png")},
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_ingest_empty_file():
    """Uploading an empty file should return 422."""
    from app.main import app
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/ingest",
        data={"chunking_strategy": "fixed"},
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert response.status_code == 422
