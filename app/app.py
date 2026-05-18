import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')


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
    
    return movies['title'].iloc[movie_indices].tolist()


st.title("Movie Recommendation System")
movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    
    recommendations = recommend(movie_name)
    
    st.write("Recommended Movies:")
    
    for movie in recommendations:
        st.write(movie)