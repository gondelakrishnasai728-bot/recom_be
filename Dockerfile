FROM python:3.10-slim

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies first (to cache this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download required NLTK datasets for textblob and training script safety
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

# Copy the application code
COPY . .

# Expose Hugging Face Space's default PORT
ENV PORT=7860
EXPOSE 7860

# Run with Gunicorn (1 worker, 2 threads to keep minimal RAM footprint)
CMD gunicorn --bind 0.0.0.0:$PORT -w 1 --threads 2 "app:app"
