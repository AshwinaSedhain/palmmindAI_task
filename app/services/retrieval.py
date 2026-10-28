from app.services.embeddings import embed_query
from app.services.vector_store import search_similar_chunks
from app.core.config import settings


def retrieve_relevant_chunks(query: str, top_k: int = None):
    if top_k is None:
        top_k = settings.top_k_results
    
    query_vec = embed_query(query)
    chunks = search_similar_chunks(query_vec, top_k=top_k)
    return chunks
