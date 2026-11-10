from app.pdf_utils import extract_pages
from app.chunking import chunk_text
from app.nlp_features import get_keywords, summarize, get_entities

with open("samples/test.pdf", "rb") as f:
    pages = extract_pages(f.read())
chunks = chunk_text(pages)

print("KEYWORDS:", get_keywords(chunks))
print()
print("SUMMARY:")
for s in summarize(chunks):
    print(f"- (page {s['page']}) {s['text']}")
print()
print("ENTITIES:", get_entities(chunks))