from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "doc_classifier.joblib"
_clf = None


def classify_document(chunks):
    """Predict the document type by averaging predictions over all chunks."""
    global _clf
    if _clf is None:
        if not MODEL_PATH.exists():
            return None
        _clf = joblib.load(MODEL_PATH)

    probs = _clf.predict_proba([c["text"] for c in chunks]).mean(axis=0)
    best = int(np.argmax(probs))
    return {"type": str(_clf.classes_[best]), "confidence": float(probs[best])}