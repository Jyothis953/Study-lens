import fitz  # this is PyMuPDF


def extract_pages(pdf_bytes):
    """Read a PDF and return a list of (page_number, text) pairs."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append((i + 1, text))
    doc.close()
    return pages