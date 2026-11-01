def chunk_text(pages, chunk_size=150, overlap=30):
    """Split (page_number, text) pairs into small overlapping chunks."""
    chunks = []
    step = chunk_size - overlap   # how far we move forward each time

    for page_num, text in pages:
        words = text.split()      # break the page into a list of words

        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            if piece.strip():     # skip empty pieces
                chunks.append({"page": page_num, "text": piece})

    return chunks