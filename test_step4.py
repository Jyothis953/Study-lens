from app.pdf_utils import extract_pages
from app.chunking import chunk_text

with open("samples/test.pdf", "rb") as f:
    data = f.read()

pages = extract_pages(data)
chunks = chunk_text(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))
print("-" * 40)
print("First chunk:")
print("Page:", chunks[0]["page"])
print("Text:", chunks[0]["text"])