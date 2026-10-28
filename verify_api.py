#!/usr/bin/env python3
"""Quick API verification script"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("1. Testing health endpoint...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("   PASS\n")

def test_ingestion():
    print("2. Testing document ingestion...")
    
    # Create test file
    content = b"Python was created by Guido van Rossum. It is a high-level programming language."
    files = {"file": ("test.txt", content, "text/plain")}
    data = {
        "chunking_strategy": "fixed",
        "chunk_size": "50",
        "chunk_overlap": "10"
    }
    
    resp = requests.post(f"{BASE_URL}/api/v1/documents/ingest", files=files, data=data)
    print(f"   Status: {resp.status_code}")
    result = resp.json()
    print(f"   Document ID: {result['document_id']}")
    print(f"   Chunks: {result['chunk_count']}")
    assert resp.status_code == 201
    assert result['chunk_count'] > 0
    print("   PASS\n")
    return result['document_id']

def test_chat(doc_id):
    print("3. Testing RAG chat...")
    
    payload = {
        "session_id": "verify-test",
        "message": "Who created Python?"
    }
    
    resp = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {resp.status_code}")
    result = resp.json()
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Sources used: {result['sources_used']}")
    assert resp.status_code == 200
    assert "answer" in result
    print("   PASS\n")

def test_booking():
    print("4. Testing interview booking...")
    
    # Start booking
    resp1 = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={"session_id": "book-verify", "message": "I want to book an interview"}
    )
    print(f"   Step 1 - Status: {resp1.status_code}")
    result1 = resp1.json()
    print(f"   Response: {result1['answer'][:80]}...")
    assert "name" in result1['answer'].lower()
    
    # Provide name
    resp2 = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json={"session_id": "book-verify", "message": "My name is Test User"}
    )
    result2 = resp2.json()
    print(f"   Step 2 - Response: {result2['answer'][:80]}...")
    assert "email" in result2['answer'].lower()
    
    print("   PASS\n")

if __name__ == "__main__":
    try:
        test_health()
        doc_id = test_ingestion()
        test_chat(doc_id)
        test_booking()
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
