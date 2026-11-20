from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from app.pdf_utils import extract_pages
from app.chunking import chunk_text

DATA_DIR = Path("training_data")
MODEL_PATH = Path("models") / "doc_classifier.joblib"

# 1. Load every PDF and turn it into labelled text chunks
texts, labels, groups = [], [], []
for folder in sorted(DATA_DIR.iterdir()):
    if not folder.is_dir():
        continue
    pdfs = list(folder.glob("*.pdf"))
    print(f"{folder.name}: {len(pdfs)} PDFs")
    for pdf in pdfs:
        chunks = chunk_text(extract_pages(pdf.read_bytes()))
        for c in chunks:
            texts.append(c["text"])
            labels.append(folder.name)
            groups.append(str(pdf))     # remember which file each chunk came from

print("Total training chunks:", len(texts))

# 2. Split by FILE, so the test set contains documents the model has never seen
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(texts, labels, groups))

X_train = [texts[i] for i in train_idx]
y_train = [labels[i] for i in train_idx]
X_test = [texts[i] for i in test_idx]
y_test = [labels[i] for i in test_idx]

# 3. Train: TF-IDF turns text into numbers, LogisticRegression learns the types
model = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=5000),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)

# 4. Evaluate on unseen documents
predictions = model.predict(X_test)
print()
print("Accuracy on unseen documents:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))

# 5. Retrain on ALL data and save the final model
model.fit(texts, labels)
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("Model saved to", MODEL_PATH)