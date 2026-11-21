import json

from app.pdf_utils import extract_pages
from app.chunking import chunk_text
from app.search import build_index, search

PDF_PATH = "samples/test.pdf"
QUESTIONS_PATH = "eval_questions.json"

# Load the document and build the search index
with open(PDF_PATH, "rb") as f:
    pages = extract_pages(f.read())
chunks = chunk_text(pages)
embeddings = build_index(chunks)

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)

top1_hits = 0
top3_hits = 0

for q in questions:
    correct = q["page"] if isinstance(q["page"], list) else [q["page"]]
    results = search(q["question"], chunks, embeddings, k=3)
    found_pages = [r["page"] for r in results]

    in_top1 = found_pages[0] in correct
    in_top3 = any(p in correct for p in found_pages)
    top1_hits += in_top1
    top3_hits += in_top3

    mark = "OK  " if in_top3 else "MISS"
    print(f"{mark} | {q['question']}")
    print(f"       correct: {correct} | found: {found_pages}")

total = len(questions)
print()
print("Questions tested:   ", total)
print(f"Top-1 accuracy:      {top1_hits}/{total} = {top1_hits / total:.0%}")
print(f"Top-3 accuracy:      {top3_hits}/{total} = {top3_hits / total:.0%}")