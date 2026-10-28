from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        # Use in-memory Qdrant if running locally, otherwise connect to server
        try:
            _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            # Test connection
            _client.get_collections()
        except Exception:
            # Fallback to in-memory mode if server not available
            print("Qdrant server not found, using in-memory mode")
            _client = QdrantClient(":memory:")
    return _client


def ensure_collection_exists():
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    
    if settings.qdrant_collection_name not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch")
    
    client = get_client()
    points = []
    
    for idx, (text, vec) in enumerate(zip(chunks, embeddings)):
        point_id = abs(hash(f"{doc_id}:{idx}")) % (2**63)
        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "text": text,
                }
            )
        )
    
    client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
    )


def search_similar_chunks(query_vec: list[float], top_k: int):
    client = get_client()
    results = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )
    
    return [
        {
            "text": hit.payload.get("text", ""),
            "document_id": hit.payload.get("document_id", ""),
            "score": hit.score,
        }
        for hit in results
    ]
