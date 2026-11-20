import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_topics(chunks, embeddings, n_topics=4):
    """Group chunks into topics and describe each topic with keywords."""
    n = min(n_topics, len(chunks))
    if n < 2:
        return []

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    topics = []
    for t in range(n):
        idx = np.where(labels == t)[0]
        scores = np.asarray(matrix[idx].sum(axis=0)).ravel()
        top = scores.argsort()[::-1][:5]
        topics.append({
            "keywords": [str(words[i]) for i in top],
            "size": int(len(idx)),
            "pages": sorted({chunks[i]["page"] for i in idx}),
        })

    return sorted(topics, key=lambda x: -x["size"])