"""
Movie recommendation engine for the Netflix Churn & CLV project.

Recommendation logic:
1. Read the user's preferred genre from the users table.
2. Keep movies belonging to that genre.
3. Use TF-IDF on genre + keywords + overview.
4. Rank matching movies using content similarity and movie rating.
5. Return the top N movies.

No watch history is required.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_FILE = os.path.join(BASE_DIR, "recommendation", "movies.csv")

_movies = None
_tfidf = None
_tfidf_matrix = None


def _load_engine():
    """Load the movie dataset and build the TF-IDF matrix once."""
    global _movies, _tfidf, _tfidf_matrix

    if _movies is not None:
        return _movies, _tfidf, _tfidf_matrix

    _movies = pd.read_csv(MOVIES_FILE)

    for col in ["genres", "overview", "keywords"]:
        _movies[col] = _movies[col].fillna("").astype(str)

    _movies["content"] = (
        _movies["genres"] + " " +
        _movies["keywords"] + " " +
        _movies["overview"]
    )

    _tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    _tfidf_matrix = _tfidf.fit_transform(_movies["content"])

    return _movies, _tfidf, _tfidf_matrix


def recommend_movies(preferred_genre, top_n=5):
    """
    Return top movies for a user's preferred genre.

    The preferred genre is used as the primary filter.
    Within that genre, movies are ranked using:
      - content similarity to the genre profile
      - movie rating

    Returns a list of plain dictionaries for Jinja.
    """
    movies, tfidf, tfidf_matrix = _load_engine()

    preferred_genre = str(preferred_genre or "").strip()

    if not preferred_genre:
        preferred_genre = "Action"

    # Primary filter: user's preferred genre.
    genre_mask = movies["genres"].str.contains(
        preferred_genre,
        case=False,
        na=False,
        regex=False
    )

    candidates = movies.loc[genre_mask].copy()

    # Safety fallback if a new genre is added to the database
    # but is not yet present in the movie CSV.
    if candidates.empty:
        candidates = movies.copy()

    candidate_indices = candidates.index.tolist()

    # Create a TF-IDF vector for the preferred genre.
    genre_vector = tfidf.transform([preferred_genre])

    # Compare the user's genre with every movie's content.
    similarities = cosine_similarity(
        genre_vector,
        tfidf_matrix[candidate_indices]
    ).flatten()

    candidates["similarity"] = similarities

    # Normalize rating to 0-1 and combine it with content similarity.
    candidates["rating_score"] = candidates["rating"].astype(float) / 10.0

    # Every candidate already matches the user's preferred genre.
    # Content similarity and rating decide the ranking inside that genre.
    candidates["recommendation_score"] = (
        0.65 +
        0.25 * candidates["similarity"] +
        0.10 * candidates["rating_score"]
    )

    candidates = candidates.sort_values(
        ["recommendation_score", "rating"],
        ascending=False
    ).head(top_n)

    recommendations = []

    for _, movie in candidates.iterrows():
        score = float(movie["recommendation_score"])

        # Keep the displayed match score in a natural range.
        match_percent = round(min(99, max(65, score * 100)), 1)

        recommendations.append({
            "movie_id": int(movie["movie_id"]),
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "overview": str(movie["overview"]),
            "rating": float(movie["rating"]),
            "match_percent": match_percent,
        })

    return recommendations
