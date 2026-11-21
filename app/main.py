from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.pdf_utils import extract_pages
from app.chunking import chunk_text
from app.search import build_index, search
from app.qa import get_answer
from app.nlp_features import get_keywords, summarize, get_entities
from app.topics import cluster_topics
from app.classifier import classify_document
from app.database import (
    init_db, save_document, list_documents, load_document, delete_document,
)

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="StudyLens")
init_db()


def get_doc(doc_id: int):
    """Load a document from the database or raise a clear error."""
    loaded = load_document(doc_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return loaded


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    pages = extract_pages(data)
    chunks = chunk_text(pages)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text found. This PDF may be a scanned image.",
        )

    embeddings = build_index(chunks)
    doc_id = save_document(file.filename, len(pages), chunks, embeddings)

    return {
        "id": doc_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.get("/documents")
def documents():
    return list_documents()


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: int):
    delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/ask")
def ask(question: str, doc_id: int):
    chunks, embeddings = get_doc(doc_id)
    hits = search(question, chunks, embeddings)
    return get_answer(question, hits)


@app.get("/insights")
def insights(doc_id: int):
    chunks, _ = get_doc(doc_id)
    return {
        "keywords": get_keywords(chunks),
        "summary": summarize(chunks),
        "entities": get_entities(chunks),
    }


@app.get("/topics")
def topics(doc_id: int, n: int = 4):
    chunks, embeddings = get_doc(doc_id)
    return cluster_topics(chunks, embeddings, n)


@app.get("/doc-type")
def doc_type(doc_id: int):
    chunks, _ = get_doc(doc_id)
    result = classify_document(chunks)
    if result is None:
        raise HTTPException(status_code=400, detail="Classifier not trained yet.")
    return result