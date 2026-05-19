import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "2a94e9bef9ac248db37a5a1800fd4f92"

def fetch_poster(tmdb_id):

    try:

        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}"

        response = requests.get(url, timeout=10)

        data = response.json()
        print(data)

        poster_path = data.get('poster_path')

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"

    except:
        return None

    return None
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
links = pd.read_csv('data/links.csv')
movies = movies.merge(links, on='movieId')

movies['genres'] = movies['genres'].fillna('')


tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies['genres'])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_title):
    
    if movie_title not in indices:
        return ["Movie not found"]
    
    idx = indices[movie_title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movies = []
    recommended_posters = []

    for i in movie_indices:

        recommended_movies.append(movies.iloc[i].title)

        poster = fetch_poster(movies.iloc[i].tmdbId)

        recommended_posters.append(poster)
    return recommended_movies, recommended_posters  


st.title("Movie Recommendation System")
movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    
    recommended_movies, recommended_posters = recommend(movie_name)

    for movie, poster in zip(recommended_movies, recommended_posters):

        st.subheader(movie)

        if poster:
            st.image(poster, width=200)
        else:
            st.write("Poster not available")

    if poster:
        st.image(poster)