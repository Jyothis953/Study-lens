import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "studylens.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the tables if they don't exist yet."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            pages INTEGER NOT NULL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    conn.commit()
    conn.close()


def save_document(filename, pages, chunks, embeddings):
    """Save a document with its chunks and embeddings. Returns the new document id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO documents (filename, pages) VALUES (?, ?)", (filename, pages)
    )
    doc_id = cur.lastrowid
    rows = [
        (doc_id, c["page"], c["text"], np.asarray(e, dtype=np.float32).tobytes())
        for c, e in zip(chunks, embeddings)
    ]
    conn.executemany(
        "INSERT INTO chunks (document_id, page, text, embedding) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return doc_id


def list_documents():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, filename, pages, uploaded_at FROM documents ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_document(doc_id):
    """Return (chunks, embeddings) for a document, or None if it doesn't exist."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT page, text, embedding FROM chunks WHERE document_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    chunks = [{"page": r["page"], "text": r["text"]} for r in rows]
    embeddings = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return chunks, embeddings


def delete_document(doc_id):
    conn = get_connection()
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()