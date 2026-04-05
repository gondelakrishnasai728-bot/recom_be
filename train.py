"""
train.py - Model Training Script
=================================
This script reads goodreads_data.xlsx, runs the full NLP pipeline,
fits a TF-IDF vectorizer, computes the cosine similarity matrix,
and saves all artifacts to disk so app.py can load them instantly.

Run this once (or whenever the dataset changes):
    python backend/train.py
"""

import os
import sys
import string
import numpy as np
import pandas as pd
import joblib
import nltk
from textblob import TextBlob

# ─── NLTK Setup ──────────────────────────────────────────────────────────────
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

NLTK_PACKAGES = [
    ('corpora/stopwords',          'stopwords'),
    ('tokenizers/punkt',           'punkt'),
    ('tokenizers/punkt_tab',       'punkt_tab'),
    ('corpora/wordnet',            'wordnet'),
    ('corpora/omw-1.4',            'omw-1.4'),
]
for resource_path, package_name in NLTK_PACKAGES:
    try:
        nltk.data.find(resource_path, paths=[nltk_data_dir])
    except LookupError:
        print(f"[NLTK] Downloading '{package_name}'…")
        nltk.download(package_name, download_dir=nltk_data_dir, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(BASE_DIR, 'goodreads_data.xlsx')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
COSINE_SIM_PATH = os.path.join(MODEL_DIR, 'cosine_sim.npy')
DATAFRAME_PATH  = os.path.join(MODEL_DIR, 'books_df.pkl')

# ─── NLP Tools ────────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Full NLP pipeline:
      1. Lowercase
      2. Remove punctuation
      3. Tokenize
      4. Remove stopwords
      5. Lemmatize
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize
    tokens = word_tokenize(text)
    # 4 & 5. Remove stopwords + Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# ─── Load Data ────────────────────────────────────────────────────────────────
print(f"\n📂 Loading dataset from: {DATASET}")
df = pd.read_excel(DATASET)
print(f"   ✔ Loaded {len(df)} rows | Columns: {df.columns.tolist()}")

# Normalise column names to lower-case stripped versions
df.columns = df.columns.str.strip()

# Keep only what we need and drop rows with no description
needed = ['Book', 'Author', 'Description', 'Genres', 'Avg_Rating', 'Num_Ratings', 'URL']
for col in needed:
    if col not in df.columns:
        df[col] = ''           # create missing columns as empty

df = df[needed].copy()
df.dropna(subset=['Description'], inplace=True)
df['Description'] = df['Description'].astype(str)
df.reset_index(drop=True, inplace=True)
print(f"   ✔ After cleaning: {len(df)} rows")

# ─── NLP Pipeline ────────────────────────────────────────────────────────────
print("\n🔤 Running NLP pipeline on descriptions…")
df['processed'] = df['Description'].apply(preprocess_text)
print("   ✔ Preprocessing complete")

# ─── TF-IDF Vectorisation ────────────────────────────────────────────────────
print("\n📐 Fitting TF-IDF vectorizer…")
vectorizer = TfidfVectorizer(
    max_features=10_000,   # cap vocabulary to top-10k terms
    ngram_range=(1, 2),    # use unigrams and bigrams
    min_df=2,              # ignore very rare terms
    sublinear_tf=True,     # apply log normalization to TF
)
tfidf_matrix = vectorizer.fit_transform(df['processed'])
print(f"   ✔ TF-IDF matrix shape: {tfidf_matrix.shape}")

# ─── Cosine Similarity Matrix ─────────────────────────────────────────────────
print("\n⚙️  Computing cosine similarity matrix… (this may take a moment)")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"   ✔ Similarity matrix shape: {cosine_sim.shape}")

# ─── Pre-compute Sentiment Scores ────────────────────────────────────────────
# Computing TextBlob polarity here once avoids expensive per-request calls.
print("\n😊 Pre-computing sentiment scores…")
def get_sentiment(text: str) -> float:
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0

df['sentiment'] = df['Description'].apply(get_sentiment)
sentiment_arr   = df['sentiment'].values.astype('float32')
SENTIMENT_PATH  = os.path.join(MODEL_DIR, 'sentiment_scores.npy')
np.save(SENTIMENT_PATH, sentiment_arr)
print(f"   ✔ Sentiment scores → {SENTIMENT_PATH}")

# ─── Save Artifacts ───────────────────────────────────────────────────────────
print("\n💾 Saving model artifacts…")
joblib.dump(vectorizer, VECTORIZER_PATH)
np.save(COSINE_SIM_PATH, cosine_sim)
df.to_pickle(DATAFRAME_PATH)

print(f"   ✔ Vectorizer  → {VECTORIZER_PATH}")
print(f"   ✔ Cosine sim  → {COSINE_SIM_PATH}")
print(f"   ✔ DataFrame   → {DATAFRAME_PATH}")
print("\n✅ Training complete! You can now start app.py.\n")
