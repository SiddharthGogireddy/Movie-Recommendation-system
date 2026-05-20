from fastapi import FastAPI
from fastapi import FastAPI
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = FastAPI()
movies = pd.read_csv('data/movies.csv')
tags = pd.read_csv('data/tags.csv')
links = pd.read_csv('data/links.csv')
movies = movies.merge(links, on='movieId')

tag_data = tags.groupby('movieId')['tag'].apply(
    lambda x: " ".join(x.astype(str))
)

movies = movies.merge(
    tag_data,
    on='movieId',
    how='left'
)
movies['genres'] = movies['genres'].fillna('')
movies['tag'] = movies['tag'].fillna('')

movies['content'] = (
    movies['genres'] + " " +
    movies['genres'] + " " +
    movies['genres'] + " " +
    movies['tag']
)
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    movies['content']
)

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)
indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()
def recommend(movie_title):

    if movie_title not in indices:
        return []

    idx = indices[movie_title]

    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:11]

    movie_indices = [
        i[0]
        for i in sim_scores
    ]

    recommendations = movies[
        ['title', 'genres']
    ].iloc[movie_indices]

    return recommendations.to_dict(
        orient='records'
    )
@app.get("/recommend/{movie_title}")
def get_recommendations(movie_title: str):
    print(f"Received request for: {movie_title}")
    recommendations = recommend(movie_title)

    return {
        "movie": movie_title,
        "recommendations": recommendations
    }


