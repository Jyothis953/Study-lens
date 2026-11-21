from app.chunking import chunk_text


def test_chunk_text_creates_chunks_with_page_numbers():
    pages = [(1, "word " * 400)]
    chunks = chunk_text(pages, chunk_size=150, overlap=30)
    assert len(chunks) > 1
    assert all(c["page"] == 1 for c in chunks)


def test_chunk_text_skips_empty_pages():
    assert chunk_text([(1, "   ")]) == []


def test_chunks_never_exceed_chunk_size():
    pages = [(1, "word " * 500)]
    for c in chunk_text(pages, chunk_size=100, overlap=20):
        assert len(c["text"].split()) <= 100