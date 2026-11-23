FROM python:3.11-slim

# Hugging Face runs apps as a normal user, so we create one
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install libraries
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Copy your project
COPY --chown=user . .

# Download the AI models now, so visitors don't wait later
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "from transformers import pipeline; pipeline('question-answering', model='distilbert-base-cased-distilled-squad')"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]