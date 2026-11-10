import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.search import model  # reuse the embedding model from Step 5

_nlp = None  # spaCy is loaded only when first needed


def get_keywords(chunks, top_n=10):
    """Find the most important words in the document."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    matrix = vectorizer.fit_transform(texts)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names_out()
    top = scores.argsort()[::-1][:top_n]
    return [str(words[i]) for i in top]


def summarize(chunks, n_sentences=5):
    """Pick the sentences that best represent the whole document."""
    seen = set()
    sentences = []
    for c in chunks:
        for s in re.split(r"(?<=[.!?])\s+", c["text"]):
            s = s.strip()
            if len(s.split()) >= 6 and s not in seen:
                seen.add(s)
                sentences.append({"page": c["page"], "text": s})

    if not sentences:
        return []

    emb = model.encode([s["text"] for s in sentences], normalize_embeddings=True)
    centroid = emb.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    scores = emb @ centroid                      # closeness to the document's overall meaning

    top = sorted(np.argsort(scores)[::-1][:n_sentences])   # keep original order
    return [sentences[int(i)] for i in top]


def get_entities(chunks, per_label=8):
    """Find people, organisations, places, and dates."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")

    wanted = {"PERSON", "ORG", "GPE", "DATE"}
    counts = defaultdict(Counter)
    for c in chunks:
        for ent in _nlp(c["text"]).ents:
            if ent.label_ in wanted:
                counts[ent.label_][ent.text.strip()] += 1

    return {
        label: [word for word, _ in counter.most_common(per_label)]
        for label, counter in counts.items()
    }