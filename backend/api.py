import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, SkipValidation, ConfigDict
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import tensorflow as tf
import pickle
import os
from typing import List, Optional, Dict
import requests
from dotenv import load_dotenv
from datetime import timezone
import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specific React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Modelos Pydantic
class ModelConfig:
    arbitrary_types_allowed = True

class Movie(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "title": "Example Movie",
                "tags": ["action", "adventure"],
                "rating": 4.5,
                "predicted_rating": None,
                "tmdb_id": "123",
                "poster_path": None,
                "overview": None
            }
        }
    )
    
    id: int
    title: str
    tags: List[str] = []
    rating: Optional[float] = None
    predicted_rating: Optional[float] = None
    tmdb_id: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None

class Rating(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_id: int
    movie_id: int
    rating: float
    
class MovieRecommendation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    movie_id: int
    predicted_rating: float

class UserMovieInteraction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_id: int
    movie_id: int
    rating: Optional[float] = None
    feedback_type: Optional[str] = None
    timestamp: SkipValidation[datetime]

class Feedback(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_id: int
    movie_id: int
    feedback_type: str
    timestamp: SkipValidation[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    

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
        try:
            model_path = os.path.join(BASE_DIR, 'ml', 'models', 'model_ok', 'movie_recommender_model.keras')
            self.model = tf.keras.models.load_model(model_path)
            
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            raise Exception("No se pudo cargar el modelo")

        try:
            models_dir = os.path.join(BASE_DIR, 'ml', 'models', 'model_ok')
            with open(os.path.join(models_dir, 'user_encoder.pkl'), 'rb') as f:
                self.user_encoder = pickle.load(f)
            with open(os.path.join(models_dir, 'movie_encoder.pkl'), 'rb') as f:
                self.movie_encoder = pickle.load(f)
            with open(os.path.join(models_dir, 'tag_scaler.pkl'), 'rb') as f:
                self.tag_scaler = pickle.load(f)
        except Exception as e:
            print(f"Error al cargar los encoders: {str(e)}")
            raise

        try:
            data_path = os.path.join(BASE_DIR, 'data', 'movielens', 'link.csv')
            self.movie_links = pd.read_csv(data_path)
            self.movie_links.set_index('movieId', inplace=True)
        except Exception as e:
            print(f"Error al cargar movie_links: {str(e)}")
            raise
        
    def predict_rating(self, user_id: int, movie_id: int, neo4j_session):
        # Obtener features de tags
        tags = self.get_movie_tags(movie_id, neo4j_session)
        tag_features = self.tag_scaler.transform([list(tags.values())])

        # Codificar usuario y película
        user_encoded = self.user_encoder.transform([user_id])
        movie_encoded = self.movie_encoder.transform([movie_id])

        # Predecir rating
        prediction = self.model.predict(
            [
                user_encoded,
                movie_encoded,
                tag_features
            ],
            verbose=0  # Suprimir la salida de progreso
        )
        
        return float(prediction[0][0])


    def get_movie_tags(self, movie_id: int, neo4j_session):
        query = """
        MATCH (m:Movie {movieId: $movie_id})-[r:HAS_TAG]->(t:Tag)
        RETURN t.name as tag_name, r.relevance as relevance
        """
        result = neo4j_session.run(query, movie_id=movie_id)
        return {row['tag_name']: row['relevance'] for row in result}

    

    def get_tmdb_info(self, movie_id: int) -> dict:
        try:
            tmdb_id = self.movie_links.loc[movie_id, 'tmdbId']
            print(f"TMDB ID for movie {movie_id}: {tmdb_id}")  # Log 1
            
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={'api_key': os.getenv('TMDB_API_KEY')}
            )
            print(f"TMDB API Response: {response.status_code}")  # Log 2
            if response.status_code == 200:
                data = response.json()
                result = {
                    'poster_path': f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get('poster_path') else None,
                    'overview': data.get('overview')
                }
                print(f"TMDB Info Result: {result}")  # Log 3
                return result
        except Exception as e:
            print(f"Error in get_tmdb_info: {str(e)}")  # Log 4
        return {'poster_path': None, 'overview': None}



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
@app.get("/api/movies/popular")
async def get_popular_movies(limit: int = 10, neo4j_session = Depends(get_neo4j)):
    print("Debug: Iniciando petición popular movies")
    try:
        query = """
        MATCH (m:Movie)<-[r:RATED]-()
        WITH m, avg(r.rating) as avg_rating, count(r) as num_ratings
        RETURN m.movieId as movieId, 
               m.title as title,
               avg_rating,
               num_ratings
        LIMIT $limit
        """
        
        print("Debug: Ejecutando query")
        result = neo4j_session.run(query, limit=limit)
        movies = list(result)
        recommender = MovieRecommender()
        movies_with_info = []
        for movie in movies:
            movie_data = {"id": movie["movieId"], "title": movie["title"]}
            tmdb_info = recommender.get_tmdb_info(movie["movieId"])
            print(f"TMDB info for movie {movie['movieId']}: {tmdb_info}")  # Log debug
            movie_data.update(tmdb_info)
            movies_with_info.append(movie_data)
        
        return movies_with_info
        
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
        
        tmdb_info = recommender.get_tmdb_info(movie_id)
        movie.update(tmdb_info)
        
        movies.append(movie)
    
    return movies

# 2. Luego las rutas con parámetros de recomendaciones (más específicas que movie_id)
@app.get("/api/movies/recommendations/{user_id}", response_model=List[Movie])
async def get_recommendations(
    user_id: int,
    limit: int = 10,
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    query = """
    MATCH (u:User {userId: $user_id})
    MATCH (m:Movie)
    WHERE NOT EXISTS((u)-[:RATED]->(m))
    AND NOT EXISTS((u)-[:GAVE_FEEDBACK {type: 'not_interested'}]->(m))
    AND NOT EXISTS((u)-[:GAVE_FEEDBACK {type: 'dislike'}]->(m))
    
    OPTIONAL MATCH (u)-[p:PREFERS]->(t:Tag)<-[:HAS_TAG]-(m)
    WITH m, sum(coalesce(p.weight, 0)) as tag_score
    
    RETURN m.movieId as movieId, 
           m.title as title,
           tag_score
    ORDER BY tag_score DESC
    LIMIT 100
    """
    
    result = neo4j_session.run(query, user_id=user_id)
    movies = []
    
    for row in result:
        movie = {
            "id": row["movieId"],
            "title": row["title"]
        }
        
        # Predecir rating
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
        
        movies.append(movie)
    
    # Ordenar por predicción y limitar resultados
    movies.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return movies[:limit]

@app.post("/api/movies/recommendations/{user_id}")
async def get_personalized_recommendations(
    user_id: int,
    body: dict,  # Cambiamos esto temporalmente para ver qué recibimos
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    print("Received body:", body)  # Debug print
    
    try:
        ratings = body.get('ratings', {})
        # Convertir las claves del diccionario a enteros
        ratings_converted = {int(k): float(v) for k, v in ratings.items()}
        print("Converted ratings:", ratings_converted)  # Debug print
        
        for movie_id, rating in ratings_converted.items():
            query = """
            MATCH (u:User {userId: $user_id})
            MATCH (m:Movie {movieId: $movie_id})
            MERGE (u)-[r:RATED]->(m)
            SET r.rating = $rating
            """
            neo4j_session.run(
                query,
                user_id=user_id,
                movie_id=movie_id,
                rating=rating
            )
        
        return await get_recommendations(user_id, 10, neo4j_session, recommender)
    except Exception as e:
        print("Error processing request:", str(e))  # Debug print
        raise HTTPException(status_code=400, detail=str(e))

# 3. Finalmente las rutas con parámetros genéricos
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
    
    tags_query = """
    MATCH (m:Movie {movieId: $movie_id})-[:HAS_TAG]->(t:Tag)
    RETURN t.name as tag
    """
    tags_result = neo4j_session.run(tags_query, movie_id=movie_id)
    
    movie = {
        "id": movie_data["movieId"],
        "title": movie_data["title"],
        "tags": [row["tag"] for row in tags_result]
    }
    
    tmdb_info = recommender.get_tmdb_info(movie_id)
    movie.update(tmdb_info)
    
    return movie

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
    
    feedback_stats = {
        "like": 0,
        "dislike": 0,
        "save_for_later": 0,
        "not_interested": 0
    }
    
    for item in stats["feedback_counts"]:
        feedback_stats[item["type"]] = item["count"]
    
    return feedback_stats

@app.get("/api/users")
async def get_users(neo4j_session=Depends(get_neo4j)):
    query = "MATCH (u:User) RETURN u.userId as userId"
    result = neo4j_session.run(query)
    users = [{"userId": row["userId"]} for row in result]
    return users

@app.post("/api/users")
async def create_user(neo4j_session=Depends(get_neo4j)):
    query = """
    MATCH (u:User)
    WITH max(u.userId) as maxId
    CREATE (newUser:User {userId: maxId + 1})
    RETURN newUser.userId as userId
    """
    result = neo4j_session.run(query)
    new_user = result.single()
    if not new_user:
        raise HTTPException(status_code=500, detail="Error al crear el usuario")
    return {"userId": new_user["userId"]}

