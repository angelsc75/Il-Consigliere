import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import tensorflow as tf
import pickle
import os
from typing import List, Optional
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Movie Recommender API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Origen del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class Movie(BaseModel):
    id: int
    title: str
    tags: List[str]
    rating: Optional[float]
    predicted_rating: Optional[float]
    tmdb_id: Optional[str]
    poster_path: Optional[str]
    overview: Optional[str]

class Rating(BaseModel):
    user_id: int
    movie_id: int
    rating: float

class MovieRecommendation(BaseModel):
    movie_id: int
    predicted_rating: float
    
    
class Feedback(BaseModel):
    user_id: int
    movie_id: int
    feedback_type: str  # 'like', 'dislike', 'save_for_later', 'not_interested'
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

class UserMovieInteraction(BaseModel):
    user_id: int
    movie_id: int
    rating: Optional[float]
    feedback_type: Optional[str]
    timestamp: datetime

# Configuración de Neo4j
class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    
    def close(self):
        self.driver.close()

    def get_session(self):
        return self.driver.session()

# Clase para el recomendador
class MovieRecommender:
    def __init__(self):
        # Cargar modelo y encoders
        self.model = tf.keras.models.load_model('movie_recommender_model')
        with open('user_encoder.pkl', 'rb') as f:
            self.user_encoder = pickle.load(f)
        with open('movie_encoder.pkl', 'rb') as f:
            self.movie_encoder = pickle.load(f)
        with open('tag_scaler.pkl', 'rb') as f:
            self.tag_scaler = pickle.load(f)
        
        # Cargar mapeo de IDs de TMDB
        self.movie_links = pd.read_csv('links.csv')
        self.movie_links.set_index('movieId', inplace=True)

    def get_movie_tags(self, movie_id: int, neo4j_session):
        query = """
        MATCH (m:Movie {movieId: $movie_id})-[r:HAS_TAG]->(t:Tag)
        RETURN t.name as tag_name, r.relevance as relevance
        """
        result = neo4j_session.run(query, movie_id=movie_id)
        return {row['tag_name']: row['relevance'] for row in result}

    def predict_rating(self, user_id: int, movie_id: int, neo4j_session):
        # Obtener features de tags
        tags = self.get_movie_tags(movie_id, neo4j_session)
        tag_features = self.tag_scaler.transform([list(tags.values())])

        # Codificar usuario y película
        user_encoded = self.user_encoder.transform([user_id])
        movie_encoded = self.movie_encoder.transform([movie_id])

        # Predecir rating
        prediction = self.model.predict([
            user_encoded,
            movie_encoded,
            tag_features
        ])
        
        return float(prediction[0][0])

    def get_tmdb_info(self, movie_id: int) -> dict:
        try:
            tmdb_id = self.movie_links.loc[movie_id, 'tmdbId']
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={
                    'api_key': os.getenv('TMDB_API_KEY')
                }
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'tmdb_id': str(tmdb_id),
                    'poster_path': f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get('poster_path') else None,
                    'overview': data.get('overview')
                }
        except:
            pass
        return {'tmdb_id': None, 'poster_path': None, 'overview': None}

# Dependencias
def get_neo4j():
    conn = Neo4jConnection()
    try:
        yield conn.get_session()
    finally:
        conn.close()

def get_recommender():
    return MovieRecommender()

# Endpoints
@app.get("/api/movies/recommendations/{user_id}", response_model=List[Movie])
async def get_recommendations(
    user_id: int,
    limit: int = 10,
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    # Obtener películas no vistas por el usuario
    query = """
    MATCH (m:Movie)
    WHERE NOT EXISTS((User {userId: $user_id})-[:RATED]->(m))
    RETURN m.movieId as movieId, m.title as title
    LIMIT 100
    """
    result = neo4j_session.run(query, user_id=user_id)
    movies = [{"id": row["movieId"], "title": row["title"]} for row in result]

    # Predecir ratings y ordenar por predicción
    for movie in movies:
        movie["predicted_rating"] = recommender.predict_rating(
            user_id, movie["id"], neo4j_session
        )
        
        # Obtener tags
        tags_query = """
        MATCH (m:Movie {movieId: $movie_id})-[:HAS_TAG]->(t:Tag)
        RETURN t.name as tag
        """
        tags_result = neo4j_session.run(tags_query, movie_id=movie["id"])
        movie["tags"] = [row["tag"] for row in tags_result]

        # Obtener info de TMDB
        tmdb_info = recommender.get_tmdb_info(movie["id"])
        movie.update(tmdb_info)

    # Ordenar por predicción y limitar resultados
    movies.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return movies[:limit]

@app.get("/api/movies/search", response_model=List[Movie])
async def search_movies(
    query: str,
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    cypher_query = """
    MATCH (m:Movie)
    WHERE m.title =~ $search_pattern
    RETURN m.movieId as movieId, m.title as title
    LIMIT 10
    """
    result = neo4j_session.run(
        cypher_query,
        search_pattern=f"(?i).*{query}.*"
    )
    
    movies = []
    for row in result:
        movie_id = row["movieId"]
        
        # Obtener tags
        tags_query = """
        MATCH (m:Movie {movieId: $movie_id})-[:HAS_TAG]->(t:Tag)
        RETURN t.name as tag
        """
        tags_result = neo4j_session.run(tags_query, movie_id=movie_id)
        
        movie = {
            "id": movie_id,
            "title": row["title"],
            "tags": [r["tag"] for r in tags_result]
        }
        
        # Obtener info de TMDB
        tmdb_info = recommender.get_tmdb_info(movie_id)
        movie.update(tmdb_info)
        
        movies.append(movie)
    
    return movies

@app.post("/api/ratings")
async def add_rating(
    rating: Rating,
    neo4j_session = Depends(get_neo4j)
):
    query = """
    MATCH (u:User {userId: $user_id})
    MATCH (m:Movie {movieId: $movie_id})
    MERGE (u)-[r:RATED]->(m)
    SET r.rating = $rating
    RETURN r.rating
    """
    
    result = neo4j_session.run(
        query,
        user_id=rating.user_id,
        movie_id=rating.movie_id,
        rating=rating.rating
    )
    
    if not result.single():
        raise HTTPException(status_code=404, message="Usuario o película no encontrados")
    
    return {"status": "success"}

@app.get("/api/movies/{movie_id}", response_model=Movie)
async def get_movie(
    movie_id: int,
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    query = """
    MATCH (m:Movie {movieId: $movie_id})
    RETURN m.movieId as movieId, m.title as title
    """
    result = neo4j_session.run(query, movie_id=movie_id)
    movie_data = result.single()
    
    if not movie_data:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    # Obtener tags
    tags_query = """
    MATCH (m:Movie {movieId: $movie_id})-[:HAS_TAG]->(t:Tag)
    RETURN t.name as tag
    """
    tags_result = neo4j_session.run(tags_query, movie_id=movie_id)
    
    # Construir respuesta
    movie = {
        "id": movie_data["movieId"],
        "title": movie_data["title"],
        "tags": [row["tag"] for row in tags_result]
    }
    
    # Añadir info de TMDB
    tmdb_info = recommender.get_tmdb_info(movie_id)
    movie.update(tmdb_info)
    
    return movie
@app.post("/api/feedback")
async def add_feedback(
    feedback: Feedback,
    neo4j_session = Depends(get_neo4j)
):
    # Crear relación de feedback en Neo4j
    query = """
    MATCH (u:User {userId: $user_id})
    MATCH (m:Movie {movieId: $movie_id})
    MERGE (u)-[f:GAVE_FEEDBACK]->(m)
    SET f.type = $feedback_type,
        f.timestamp = datetime($timestamp)
    RETURN f
    """
    
    result = neo4j_session.run(
        query,
        user_id=feedback.user_id,
        movie_id=feedback.movie_id,
        feedback_type=feedback.feedback_type,
        timestamp=feedback.timestamp.isoformat()
    )
    
    if not result.single():
        raise HTTPException(status_code=404, detail="Usuario o película no encontrados")
    
    # Actualizar recomendaciones basadas en feedback
    if feedback.feedback_type in ['like', 'dislike']:
        update_recommendations_query = """
        MATCH (u:User {userId: $user_id})-[f:GAVE_FEEDBACK {type: $feedback_type}]->(m:Movie)
        MATCH (m)-[:HAS_TAG]->(t:Tag)
        WITH u, t, COUNT(*) as weight
        MERGE (u)-[p:PREFERS]->(t)
        ON CREATE SET p.weight = weight
        ON MATCH SET p.weight = p.weight + weight
        """
        
        neo4j_session.run(
            update_recommendations_query,
            user_id=feedback.user_id,
            feedback_type=feedback.feedback_type
        )
    
    return {"status": "success"}

@app.get("/api/users/{user_id}/interactions", response_model=List[UserMovieInteraction])
async def get_user_interactions(
    user_id: int,
    neo4j_session = Depends(get_neo4j)
):
    query = """
    MATCH (u:User {userId: $user_id})-[r]->(m:Movie)
    WHERE type(r) IN ['RATED', 'GAVE_FEEDBACK']
    RETURN m.movieId as movie_id, 
           m.title as title,
           CASE type(r)
               WHEN 'RATED' THEN r.rating
               ELSE null
           END as rating,
           CASE type(r)
               WHEN 'GAVE_FEEDBACK' THEN r.type
               ELSE null
           END as feedback_type,
           CASE
               WHEN r.timestamp IS NOT NULL THEN r.timestamp
               ELSE datetime()
           END as timestamp
    ORDER BY timestamp DESC
    """
    
    result = neo4j_session.run(query, user_id=user_id)
    return [dict(row) for row in result]

@app.get("/api/movies/{movie_id}/feedback-stats")
async def get_movie_feedback_stats(
    movie_id: int,
    neo4j_session = Depends(get_neo4j)
):
    query = """
    MATCH (m:Movie {movieId: $movie_id})<-[f:GAVE_FEEDBACK]-()
    WITH m, f.type as feedback_type, COUNT(*) as count
    RETURN collect({type: feedback_type, count: count}) as feedback_counts
    """
    
    result = neo4j_session.run(query, movie_id=movie_id)
    stats = result.single()
    
    if not stats:
        return {"feedback_counts": {}}
    
    # Formatear estadísticas
    feedback_stats = {
        "like": 0,
        "dislike": 0,
        "save_for_later": 0,
        "not_interested": 0
    }
    
    for item in stats["feedback_counts"]:
        feedback_stats[item["type"]] = item["count"]
    
    return feedback_stats

# Modificar el endpoint de recomendaciones existente para usar el feedback
@app.get("/api/movies/recommendations/{user_id}", response_model=List[Movie])
async def get_recommendations(
    user_id: int,
    limit: int = 10,
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    # Obtener películas no vistas y sin feedback negativo
    query = """
    MATCH (m:Movie)
    WHERE NOT EXISTS((User {userId: $user_id})-[:RATED]->(m))
    AND NOT EXISTS((User {userId: $user_id})-[:GAVE_FEEDBACK {type: 'not_interested'}]->(m))
    AND NOT EXISTS((User {userId: $user_id})-[:GAVE_FEEDBACK {type: 'dislike'}]->(m))
    
    // Boost para películas con tags preferidos por el usuario
    OPTIONAL MATCH (User {userId: $user_id})-[p:PREFERS]->(t:Tag)<-[:HAS_TAG]-(m)
    WITH m, sum(coalesce(p.weight, 0)) as tag_score
    
    RETURN m.movieId as movieId, 
           m.title as title,
           tag_score
    ORDER BY tag_score DESC
    LIMIT 100
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0000", port=8000)