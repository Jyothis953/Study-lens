# StudyLens
# StudyLens

An AI-powered website where you upload a PDF and ask questions about it.
Answers come with the page number where they were found.

## Features
- PDF upload and text extraction (PyMuPDF)
- Semantic search with sentence-transformers (all-MiniLM-L6-v2)
- Extractive question answering with DistilBERT
- Keywords, summary and named entities (TF-IDF, spaCy)
- Topic clustering (KMeans on chunk embeddings)
- Document-type classifier trained by me (TF-IDF + Logistic Regression)
- Web interface built with FastAPI and plain HTML/JavaScript

## How it works
PDF -> text extraction -> overlapping chunks -> embeddings -> semantic search -> answer + page number

## Results (current version)

**Retrieval (semantic search)**
- Top-1 accuracy: 67% (2 of 3 questions)
- Top-3 accuracy: 67% (2 of 3 questions)
- Note: this is a very small test set (3 questions), so the numbers are only
  indicative. A larger evaluation set is planned.

**Document-type classifier**
- Trained on 26 PDFs: 10 notes, 8 research papers, 8 resumes (1,473 text chunks)
- Accuracy on unseen documents: 90.1% (chunk level, 314 test chunks)
- Known limitation: the "resume" class has very few chunks (resumes are short),
  and the model did not recognise them in the test split (recall 0.00 on 2 chunks).
  Planned fix: more resumes and class-balanced training.

## How to run
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    uvicorn app.main:app

Then open http://127.0.0.1:8000

## Project structure
    app/          backend code (PDF reading, chunking, search, QA, NLP, ML)
    templates/    web page
    tests/        pytest tests
    evaluate.py   retrieval evaluation
    train_classifier.py   trains the document-type classifier

## Future work
- Larger evaluation set and chunk-size comparison
- Generative answers (full RAG with an LLM)
- Database storage and user login
- Docker and online deployment