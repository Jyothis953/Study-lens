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


BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="StudyLens")

# Simple memory: holds the current document while the server runs
store = {}


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

    store["chunks"] = chunks
    store["embeddings"] = build_index(chunks)

    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.get("/ask")
def ask(question: str):
    if "chunks" not in store:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    hits = search(question, store["chunks"], store["embeddings"])
    return get_answer(question, hits)


@app.get("/insights")
def insights():
    if "chunks" not in store:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    chunks = store["chunks"]
    return {
        "keywords": get_keywords(chunks),
        "summary": summarize(chunks),
        "entities": get_entities(chunks),
    }

@app.get("/topics")
def topics(n: int = 4):
    if "chunks" not in store:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    return cluster_topics(store["chunks"], store["embeddings"], n)

@app.get("/doc-type")
def doc_type():
    if "chunks" not in store:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    result = classify_document(store["chunks"])
    if result is None:
        raise HTTPException(status_code=400, detail="Classifier not trained yet.")
    return result