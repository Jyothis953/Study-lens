from app.pdf_utils import extract_pages
from app.chunking import chunk_text
from app.search import build_index
from app.topics import cluster_topics

with open("samples/test.pdf", "rb") as f:
    pages = extract_pages(f.read())
chunks = chunk_text(pages)
embeddings = build_index(chunks)

for i, t in enumerate(cluster_topics(chunks, embeddings), start=1):
    print(f"Topic {i}: {', '.join(t['keywords'])}")
    print(f"   {t['size']} chunks, pages {t['pages']}")