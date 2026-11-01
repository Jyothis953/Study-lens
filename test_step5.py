from app.pdf_utils import extract_pages
from app.chunking import chunk_text
from app.search import build_index, search

with open("samples/test.pdf", "rb") as f:
    data = f.read()

pages = extract_pages(data)
chunks = chunk_text(pages)

print("Building embeddings... (first run downloads the model)")
embeddings = build_index(chunks)
print("Embeddings shape:", embeddings.shape)

question = "What is the future scope of the project?"
results = search(question, chunks, embeddings)

print("-" * 40)
print("Question:", question)
for r in results:
    print()
    print(f"Page {r['page']} | score: {r['score']:.3f}")
    print(r["text"][:300])