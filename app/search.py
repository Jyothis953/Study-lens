from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model (downloads about 90 MB on the first run)
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks):
    """Turn every chunk's text into numbers (embeddings)."""
    texts = [c["text"] for c in chunks]
    return model.encode(texts, normalize_embeddings=True)


def search(question, chunks, embeddings, k=3):
    """Find the k chunks whose meaning is closest to the question."""
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q                  # similarity score for every chunk
    top = np.argsort(scores)[::-1][:k]       # positions of the k best scores
    return [{**chunks[i], "score": float(scores[i])} for i in top]