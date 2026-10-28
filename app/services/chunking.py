import nltk
from typing import Literal

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def chunk_fixed(text: str, size: int = 500, overlap: int = 50):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk size or overlap")
    
    chunks = []
    pos = 0
    
    while pos < len(text):
        end = pos + size
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos += size - overlap
    
    return chunks


def chunk_sentences(text: str, size: int = 500):
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        sent_len = len(sent)
        
        # handle very long sentences
        if not current and sent_len > size:
            chunks.append(sent)
            continue
        
        if current_len + sent_len + 1 > size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len + 1
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks


def chunk_text(text: str, strategy: Literal["fixed", "sentence"], 
               chunk_size: int = 500, chunk_overlap: int = 50):
    if strategy == "fixed":
        return chunk_fixed(text, size=chunk_size, overlap=chunk_overlap)
    elif strategy == "sentence":
        return chunk_sentences(text, size=chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
