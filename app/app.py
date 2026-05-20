import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
<style>

/* Main app background */
.stApp {
    background: linear-gradient(
        120deg,
        #ff7e5f,
        #feb47b,
        #ff9966
    );

    color: white;
}
/* Main title */
h1 {
    color: #E50914 !important;
    text-align: center;
    font-size: 3rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
}

/* Sidebar glass effect */
[data-testid="stSidebar"] > div:first-child {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
}

/* Dropdown styling */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Input text color */
input {
    color: white !important;
}

/* Recommend button */
div.stButton > button {
    background: linear-gradient(90deg, #E50914, #ff4b2b);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    transition: 0.3s;
}

/* Button hover */
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
}

/* Movie captions */
[data-testid="stCaptionContainer"] {
    text-align: center;
    color: white !important;
    font-size: 14px;
}

/* Poster image styling */
img {
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)
API_KEY = "2a94e9bef9ac248db37a5a1800fd4f92"
BACKEND_URL = "http://127.0.0.1:8000"

def get_recommendations(movie_title):

    url = f"{BACKEND_URL}/recommend/{movie_title}"

    response = requests.get(url)

    data = response.json()

    return data['recommendations']
def fetch_poster(tmdb_id):

    try:

        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}"

        response = requests.get(url, timeout=10)

        data = response.json()
        

        poster_path = data.get('poster_path')

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"

    except:
        return None

    return None
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
links = pd.read_csv('data/links.csv')
tags = pd.read_csv('data/tags.csv')
movies = movies.merge(links, on='movieId')


tag_data = tags.groupby('movieId')['tag'].apply(
    lambda x: " ".join(x.astype(str))
)
movies = movies.merge(tag_data, on='movieId', how='left')
movies['genres'] = movies['genres'].fillna('')
movies['tag'] = movies['tag'].fillna('')
movies['content'] = (
    movies['genres'] + " " +
    movies['genres'] + " " +
    movies['genres'] + " " +
    movies['tag']
)



indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()



st.title("Movie Recommendation System")
st.sidebar.title("About")

st.sidebar.write(
    "This recommendation system uses TF-IDF vectorization and cosine similarity to suggest similar movies."
)
movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):

    with st.spinner("Finding similar movies..."):
    
        recommendations = get_recommendations(movie_name)

        cols = st.columns(5)

    

        for idx, movie in enumerate(recommendations):

            with cols[idx % 5]:

                title = movie['title']

                st.caption(title)

                movie_row = movies[movies['title'] == title]

                if not movie_row.empty:

                    tmdb_id = movie_row.iloc[0].tmdbId

                    poster = fetch_poster(tmdb_id)

                    if poster:
                        st.image(poster)
                    else:
                        st.write("Poster not available")

    