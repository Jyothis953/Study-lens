from app.pdf_utils import extract_pages
from app.chunking import chunk_text
from app.search import build_index, search
from app.qa import get_answer

with open("samples/test.pdf", "rb") as f:
    data = f.read()

pages = extract_pages(data)
chunks = chunk_text(pages)
embeddings = build_index(chunks)

question = "What is the future scope of the project?"

hits = search(question, chunks, embeddings)
result = get_answer(question, hits)

print("Question:  ", question)
print("Answer:    ", result["answer"])
print("Confidence:", round(result["confidence"], 3))
print("Source:     Page", result["page"])
print("-" * 40)
print("Context:", result["context"][:400])