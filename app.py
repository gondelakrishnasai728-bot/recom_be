"""
app.py - Flask API for Book Recommendation System
===================================================
Loads pre-trained TF-IDF model artifacts produced by train.py and
serves recommendations via GET /recommend?book_name=<name>&n=<number>

Performance optimisations:
 - Sentiment scores are pre-computed at train time (no TextBlob at request time)
 - Scoring uses vectorised NumPy instead of a Python for-loop
 - Top-N selection uses np.argpartition (O(n)) instead of full sort (O(n log n))
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR       = os.path.join(os.path.dirname(__file__), 'model')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
COSINE_SIM_PATH = os.path.join(MODEL_DIR, 'cosine_sim.npy')
DATAFRAME_PATH  = os.path.join(MODEL_DIR, 'books_df.pkl')
SENTIMENT_PATH  = os.path.join(MODEL_DIR, 'sentiment_scores.npy')

# ─── Load Pre-trained Model Artifacts ─────────────────────────────────────────
def load_model():
    required = [VECTORIZER_PATH, COSINE_SIM_PATH, DATAFRAME_PATH, SENTIMENT_PATH]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing model artifacts – run backend/train.py first:")
        for p in missing:
            print(f"  ✗ {p}")
        return None, None, None, None

    print("[INFO] Loading pre-trained model artifacts…")
    vectorizer     = joblib.load(VECTORIZER_PATH)
    cosine_sim     = np.load(COSINE_SIM_PATH)          # shape (N, N) float32
    df             = pd.read_pickle(DATAFRAME_PATH)
    sentiment_arr  = np.load(SENTIMENT_PATH)            # shape (N,)  float32

    # Pre-clip sentiment to [0, 1] so the weighted add is always positive
    sentiment_arr  = np.clip(sentiment_arr, 0.0, 1.0).astype('float32')

    print(f"[INFO] Model ready — {len(df):,} books loaded.")
    return vectorizer, cosine_sim, df, sentiment_arr

vectorizer, cosine_sim, df, sentiment_arr = load_model()

# Build a lowercased title series once for fast look-ups
title_lower = df['Book'].str.lower() if df is not None else None

# ─── Helper Functions ────────────────────────────────────────────────────────
def _get_book_recommendations(book_idx, n):
    sim  = cosine_sim[book_idx].astype('float32')     # (N,)
    combined = 0.8 * sim + 0.2 * sentiment_arr        # element-wise, ultra-fast
    combined[book_idx] = -1.0 # Exclude the query book itself

    # Get top-n indices efficiently with argpartition
    if n < len(combined):
        top_idx = np.argpartition(combined, -n)[-n:]
        top_idx = top_idx[np.argsort(combined[top_idx])[::-1]]
    else:
        top_idx = np.argsort(combined)[::-1]

    results = []
    for idx in top_idx:
        row  = df.iloc[idx]
        desc = str(row.get('Description', ''))
        results.append({
            "title":       row['Book'],
            "author":      row.get('Author', ''),
            "description": desc[:250] + ('…' if len(desc) > 250 else ''),
            "genres":      row.get('Genres', ''),
            "avg_rating":  row.get('Avg_Rating', ''),
            "url":         row.get('URL', ''),
            "score":       round(float(combined[idx]), 4),
        })
    return jsonify(results)

def _get_genre_recommendations(matched_df, n):
    matched = matched_df.copy()
    matched['Avg_Rating']  = pd.to_numeric(matched['Avg_Rating'],  errors='coerce').fillna(0)
    matched['Num_Ratings'] = pd.to_numeric(matched['Num_Ratings'], errors='coerce').fillna(0)
    matched = matched.sort_values(['Avg_Rating', 'Num_Ratings'], ascending=[False, False])

    results = []
    for _, row in matched.head(n).iterrows():
        desc = str(row.get('Description', ''))
        results.append({
            "title":       row.get('Book', 'Unknown'),
            "author":      row.get('Author', ''),
            "description": desc[:250] + ('…' if len(desc) > 250 else ''),
            "genres":      row.get('Genres', ''),
            "avg_rating":  row.get('Avg_Rating', ''),
            "url":         row.get('URL', ''),
        })
    return jsonify(results)

# ─── API Endpoint: /recommend (Unified Search) ────────────────────────────────
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Unified search: matches exact book -> exact genre -> partial book -> partial genre.
    """
    if df is None:
        return jsonify({"error": "Model not trained yet. Run 'python backend/train.py' first."}), 503

    query = request.args.get('book_name', '').strip()
    try:
        n = max(1, int(request.args.get('n', 5)))
    except ValueError:
        n = 5

    if not query:
        return jsonify({"error": "Search query must not be empty."}), 400

    lower = query.lower()

    # 1. Exact Book Title Match
    exact_book = df[title_lower == lower]
    if not exact_book.empty:
        return _get_book_recommendations(int(exact_book.index[0]), n)

    # 2. Exact Genre Match
    genres_lower = df['Genres'].fillna('').str.lower()
    genre_mask = genres_lower.apply(lambda gs: lower in [x.strip() for x in gs.split(',')])
    if genre_mask.any():
        return _get_genre_recommendations(df[genre_mask], n)

    # 3. Partial Book Title Match
    partial_book = df[title_lower.str.contains(lower, regex=False, na=False)]
    if not partial_book.empty:
        return _get_book_recommendations(int(partial_book.index[0]), n)

    # 4. Partial Genre Match
    partial_genre = df[genres_lower.str.contains(lower, regex=False, na=False)]
    if not partial_genre.empty:
        return _get_genre_recommendations(partial_genre, n)

    return jsonify({"error": f"'{query}' not found as a book or genre."}), 404


# ─── API Endpoint: /recommend_by_genre (Multi-genre) ─────────────────────────
@app.route('/recommend_by_genre', methods=['GET'])
def recommend_by_genre():
    """
    Used primarily by the landing page to fetch recommendations matching an array of genres.
    """
    if df is None:
        return jsonify({"error": "Model not trained yet."}), 503

    genres_param = request.args.get('genres', '').strip()
    try:
        n = max(1, int(request.args.get('n', 10)))
    except ValueError:
        n = 10

    if not genres_param:
        return jsonify({"error": "genres parameter is required."}), 400

    user_genres = [g.strip().lower() for g in genres_param.split(',') if g.strip()]

    # Vectorised multi-genre partial match
    genres_lower = df['Genres'].fillna('').str.lower()
    mask = genres_lower.str.contains('|'.join(user_genres), regex=True, na=False)
    matched = df[mask]

    if matched.empty:
        return jsonify([])

    return _get_genre_recommendations(matched, n)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
