from app.pdf_utils import extract_pages

with open("samples/test.pdf", "rb") as f:
    data = f.read()

pages = extract_pages(data)

print("Total pages:", len(pages))
print("-" * 40)
print("Page 1 preview:")
print(pages[0][1][:500])