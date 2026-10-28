from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]):
    if not texts:
        return []
    
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(query: str):
    return embed_texts([query])[0]
