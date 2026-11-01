from transformers import pipeline

# Load the question-answering model (downloads about 250 MB on the first run)
qa_model = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


def get_answer(question, hits):
    """Try each top chunk and keep the answer the model is most confident about."""
    best = None

    for h in hits:
        result = qa_model(question=question, context=h["text"])

        if best is None or result["score"] > best["confidence"]:
            best = {
                "answer": result["answer"],
                "confidence": float(result["score"]),
                "page": h["page"],
                "context": h["text"],
            }

    return best