#  AI-Powered Movie Recommendation System

A full-stack AI recommendation platform that suggests movies using:
- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation System

Built using FastAPI, Streamlit, TF-IDF, Cosine Similarity, and TMDB API.

---

##  Features

- AI-powered movie recommendations
- TF-IDF + Cosine Similarity
- Semantic tag-based recommendations
- Collaborative filtering using ratings
- Hybrid recommendation engine
- FastAPI backend
- Streamlit frontend
- TMDB movie posters
- Responsive UI with custom styling

---

##  Architecture

```text
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
Recommendation Engine
        ↓
TMDB API
```

---

##  Tech Stack

### AI / ML
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

### Frontend
- Streamlit

### Backend
- FastAPI
- Uvicorn

### APIs
- TMDB API

---

##  Project Structure

```text
movie-reco/
│
├── app/
├── backend/
├── data/
├── notebooks/
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone YOUR_REPOSITORY_URL
cd movie-reco
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

### Start Backend

```bash
uvicorn backend.main:app --reload
```

### Start Frontend

```bash
streamlit run app/app.py
```

---

## 📡 API Endpoint

```text
GET /recommend/{movie_title}
```

Example:

```text
http://127.0.0.1:8000/recommend/Toy%20Story%20(1995)
```

---

##  Future Improvements

- Docker deployment
- Cloud deployment
- Authentication
- Watchlists
- Transformer-based recommendations
- Vector embeddings

---

##  What I Learned

- Recommendation systems
- NLP feature engineering
- Collaborative filtering
- Hybrid recommenders
- Frontend/backend separation
- REST APIs
- Full-stack AI architecture

---

## Final Outcome

This project evolved from a basic ML notebook into a full-stack AI-powered recommendation platform with:
- AI/ML
- Backend APIs
- Frontend UI
- External API integration
- Modular architecture