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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
            
            # Store known label ranges
            self.known_user_labels = set(self.user_encoder.classes_)
            self.min_user_id = min(self.known_user_labels)
            self.max_user_id = max(self.known_user_labels)
            
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

    def map_user_id(self, user_id):
        """
        Maps an unseen user ID to a known user ID range.
        """
        # Convert numpy int64 to regular int if necessary
        user_id = int(user_id)
        
        # If user_id is already in known labels, return it
        if user_id in self.known_user_labels:
            return user_id
            
        # Map to known range using modulo
        mapped_id = self.min_user_id + (user_id - self.min_user_id) % (self.max_user_id - self.min_user_id + 1)
        while mapped_id not in self.known_user_labels:
            mapped_id = self.min_user_id + (mapped_id - self.min_user_id + 1) % (self.max_user_id - self.min_user_id + 1)
            
        return mapped_id

    def predict_rating(self, user_id: int, movie_id: int, neo4j_session):
        """
        Predice el rating que un usuario daría a una película usando el modelo de red neuronal.
        """
        try:
            # Map user_id to known range
            mapped_user_id = self.map_user_id(user_id)
            print(f"Mapped user_id {user_id} to {mapped_user_id}")

            # Handle movie_id
            if movie_id not in self.movie_encoder.classes_:
                print(f"Warning: Movie ID {movie_id} not in training set")
                return 3.5  # Return neutral rating for unknown movies
            
            # Get tag features
            tags = self.get_movie_tags(movie_id, neo4j_session)
            tag_features_df = pd.DataFrame([tags], columns=self.tag_scaler.feature_names_in_)
            tag_features = self.tag_scaler.transform(tag_features_df)

            # Transform IDs
            try:
                user_encoded = self.user_encoder.transform([mapped_user_id])
                movie_encoded = self.movie_encoder.transform([movie_id])
            except Exception as e:
                print(f"Error encoding IDs: {str(e)}")
                return 3.5

            # Make prediction
            try:
                prediction = self.model.predict(
                    [user_encoded, movie_encoded, tag_features],
                    verbose=0
                )
                predicted_rating = float(np.clip(prediction[0][0], 0.5, 5.0))
                print(f"Raw prediction for user {user_id} (mapped: {mapped_user_id}), movie {movie_id}: {predicted_rating}")
            except Exception as e:
                print(f"Error in model prediction: {str(e)}")
                return 3.5

            # Adjust based on user history
            try:
                user_ratings_query = """
                MATCH (u:User {userId: $user_id})-[r:RATED]->()
                RETURN avg(r.rating) as avg_rating, count(r) as rating_count
                """
                result = neo4j_session.run(user_ratings_query, user_id=user_id)
                user_stats = result.single()
                
                if user_stats and user_stats["rating_count"] > 0:
                    user_avg = user_stats["avg_rating"]
                    global_avg = 3.5
                    bias = user_avg - global_avg
                    predicted_rating = np.clip(predicted_rating + (bias * 0.5), 0.5, 5.0)
                    print(f"Adjusted prediction with user bias: {predicted_rating}")
            except Exception as e:
                print(f"Error adjusting prediction: {str(e)}")
                # Continue with unadjusted prediction
            
            return predicted_rating
            
        except Exception as e:
            print(f"Error in predict_rating: {str(e)}")
            return 3.5  # Return neutral rating on error
    


    def get_movie_tags(self, movie_id: int, neo4j_session):
        """
        Obtiene los tags de una película y asegura que coincidan con el formato de entrenamiento
        """
        # Obtener todos los tags posibles que se usaron en el entrenamiento
        training_tags = set(self.tag_scaler.feature_names_in_)
        
        # Obtener los tags actuales de la película
        query = """
        MATCH (m:Movie {movieId: $movie_id})-[r:HAS_TAG]->(t:Tag)
        RETURN t.name as tag_name, r.relevance as relevance
        """
        result = neo4j_session.run(query, movie_id=movie_id)
        current_tags = {row['tag_name']: row['relevance'] for row in result}
        
        # Crear un diccionario con todos los tags del entrenamiento, con valor 0 si no existe
        tag_features = {tag: current_tags.get(tag, 0.0) for tag in training_tags}
        
        return tag_features

    

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
        # Primero obtenemos un ID aleatorio
        random_id_query = """
        MATCH (m:Movie)
        WITH min(m.movieId) as min_id, max(m.movieId) as max_id
        RETURN min_id, max_id
        """
        
        result = neo4j_session.run(random_id_query)
        range_data = result.single()
        min_id = range_data['min_id']
        max_id = range_data['max_id']
        
        import random
        random_start_id = random.randint(min_id, max_id - limit)
        
        # Ahora obtenemos las películas a partir de ese ID
        query = """
        MATCH (m:Movie)
        WHERE m.movieId >= $start_id
        WITH m
        ORDER BY m.movieId
        LIMIT $limit
        OPTIONAL MATCH (m)<-[r:RATED]-()
        RETURN m.movieId as movieId, 
               m.title as title,
               round(coalesce(AVG(r.rating), 0) * 100) / 100 as avg_rating
        """
        
        print("Debug: Ejecutando query")
        result = neo4j_session.run(query, start_id=random_start_id, limit=limit)
        movies = list(result)
        recommender = MovieRecommender()
        movies_with_info = []
        
        for movie in movies:
            movie_data = {
                "id": movie["movieId"],
                "title": movie["title"],
                "rating": movie["avg_rating"],
            }
            tmdb_info = recommender.get_tmdb_info(movie["movieId"])
            print(f"TMDB info for movie {movie['movieId']}: {tmdb_info}")
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
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    try:
        # Verify user exists
        user_query = "MATCH (u:User {userId: $user_id}) RETURN u"
        user_result = neo4j_session.run(user_query, user_id=user_id)
        user = user_result.single()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Get user's genre preferences with null check
        genre_query = """
        MATCH (u:User {userId: $user_id})-[r:RATED]->(m:Movie)
        WHERE m.genres IS NOT NULL
        UNWIND m.genres as genre
        WITH genre, avg(r.rating) as genre_rating
        WHERE genre IS NOT NULL AND genre <> '(no genres listed)'
        WITH genre, genre_rating
        ORDER BY genre_rating DESC
        LIMIT 3
        RETURN collect(genre) as preferred_genres
        """
        genre_result = neo4j_session.run(genre_query, user_id=user_id)
        genre_record = genre_result.single()
        
        # Provide default empty list if no genres found
        preferred_genres = genre_record["preferred_genres"] if genre_record and genre_record["preferred_genres"] else []

        # Modified query to handle null genres
        query = """
        MATCH (m:Movie)
        WHERE NOT EXISTS {
            MATCH (u:User {userId: $user_id})-[:RATED]->(m)
        }
        WITH m, 
             CASE 
                WHEN m.genres IS NOT NULL AND
                     any(genre IN m.genres WHERE genre IN $preferred_genres)
                THEN 2
                ELSE 1
             END as relevance_score,
             rand() as random_score
        ORDER BY relevance_score DESC, random_score
        SKIP $offset
        LIMIT $limit
        RETURN m.movieId as movieId, 
               m.title as title,
               CASE 
                   WHEN m.genres IS NULL THEN []
                   ELSE m.genres
               END as genres
        """
        
        import random
        offset = random.randint(0, 200)
        result = neo4j_session.run(
            query, 
            user_id=user_id,
            preferred_genres=preferred_genres,
            offset=offset,
            limit=50
        )
        
        candidates = []
        
        # Process results with null checks
        for row in result:
            if row is None:
                continue
                
            movie = {
                "id": row["movieId"],
                "title": row["title"],
                "tags": row["genres"] if row["genres"] else []
            }
            
            try:
                predicted_rating = recommender.predict_rating(user_id, movie["id"], neo4j_session)
                movie["predicted_rating"] = predicted_rating

                tmdb_info = recommender.get_tmdb_info(movie["id"])
                if tmdb_info:
                    movie.update(tmdb_info)
            except Exception as e:
                print(f"Error processing movie {movie['id']}: {str(e)}")
                continue

            candidates.append(movie)
        
        # Handle case when no candidates are found
        if not candidates:
            # Fallback to popular movies
            popular_query = """
            MATCH (m:Movie)<-[r:RATED]-()
            WITH m, COUNT(r) as rating_count, AVG(r.rating) as avg_rating
            WHERE rating_count > 10
            RETURN m.movieId as movieId, 
                   m.title as title,
                   m.genres as genres
            ORDER BY avg_rating DESC, rating_count DESC
            LIMIT 3
            """
            popular_result = neo4j_session.run(popular_query)
            for row in popular_result:
                movie = {
                    "id": row["movieId"],
                    "title": row["title"],
                    "tags": row["genres"] if row["genres"] else [],
                    "predicted_rating": 3.5  # Default neutral rating
                }
                tmdb_info = recommender.get_tmdb_info(movie["id"])
                if tmdb_info:
                    movie.update(tmdb_info)
                candidates.append(movie)
        
        # Sort and return recommendations with randomization
        if candidates:
            candidates.sort(key=lambda x: (x.get("predicted_rating", 0) * random.uniform(0.95, 1.05)), reverse=True)
            return candidates[:3]
        else:
            return []

    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/movies/recommendations/{user_id}")
async def get_personalized_recommendations(
    user_id: int,
    body: dict,  
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
            SET r.rating = $rating, r.timestamp = datetime()
            """
            neo4j_session.run(
                query,
                user_id=user_id,
                movie_id=movie_id,
                rating=rating
            )
        
        return await get_recommendations(user_id, neo4j_session, recommender)
    except Exception as e:
        print("Error processing request:", str(e))  # Debug print
        raise HTTPException(status_code=400, detail=str(e))


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

@app.get("/api/users/{user_id}")
async def get_user(user_id: int, neo4j_session = Depends(get_neo4j)):
    """Verifica si un usuario existe"""
    query = "MATCH (u:User {userId: $user_id}) RETURN u"
    result = neo4j_session.run(query, user_id=user_id)
    user = result.single()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"userId": user_id}

@app.get("/api/users/{user_id}/status")
async def get_user_status(user_id: int, neo4j_session = Depends(get_neo4j)):
    """Obtiene el estado del usuario: si es nuevo o tiene suficientes ratings"""
    query = """
    MATCH (u:User {userId: $user_id})-[r:RATED]->()
    RETURN count(r) as rating_count
    """
    result = neo4j_session.run(query, user_id=user_id)
    record = result.single()
    rating_count = record["rating_count"] if record else 0
    
    return {
        "hasEnoughRatings": rating_count >= 10,
        "ratingCount": rating_count
    }
@app.get("/api/users/{user_id}/status")
async def get_user_status(user_id: int, neo4j_session=Depends(get_neo4j)):
    """Gets user rating status"""
    try:
        # Check user exists
        check_query = "MATCH (u:User {userId: $user_id}) RETURN u"
        user = neo4j_session.run(check_query, user_id=user_id).single()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        # Get rating count
        count_query = """
        MATCH (u:User {userId: $user_id})-[r:RATED]->()
        RETURN COUNT(r) as rating_count
        """
        ratings = neo4j_session.run(count_query, user_id=user_id).single()
        count = ratings["rating_count"] if ratings else 0
        
        return {
            "hasEnoughRatings": count >= 10,
            "ratingCount": count
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/movies/search/combined")
async def search_movies_combined(
    query: str = None,
    tag: str = None,
    genre: str = None,
    neo4j_session = Depends(get_neo4j),
    recommender: MovieRecommender = Depends(get_recommender)
):
    """Búsqueda combinada por título, tag o género"""
    base_query = "MATCH (m:Movie)"
    where_clauses = []
    params = {}
    
    if query:
        where_clauses.append("m.title =~ $title_pattern")
        params["title_pattern"] = f"(?i).*{query}.*"
    
    if tag:
        base_query += "-[:HAS_TAG]->(t:Tag)"
        where_clauses.append("t.name = $tag")
        params["tag"] = tag
        
    if genre:
        base_query += ""
        where_clauses.append("m.genres = $genre")
        params["genre"] = genre
        
    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
        
    base_query += """
            OPTIONAL MATCH (m:Movie)<-[r:RATED]-(:User)
            RETURN m.movieId as movieId, 
                m.title as title, 
                round(coalesce(AVG(r.rating), 0) * 100) / 100 as avg_rating 
            LIMIT 20
            """
    
    result = neo4j_session.run(base_query, params)
    movies = []
    
    for row in result:
        movie_id = row["movieId"]
        movie = {
            "id": movie_id,
            "title": row["title"],
            "rating": row["avg_rating"],
        }
        # Obtener tags
        tags_query = """
        MATCH (m:Movie {movieId: $movie_id})-[:HAS_TAG]->(t:Tag)
        RETURN t.name as tag
        """
        tags_result = neo4j_session.run(tags_query, movie_id=movie_id)
        movie["tags"] = [r["tag"] for r in tags_result]
        
        # Obtener info de TMDB
        tmdb_info = recommender.get_tmdb_info(movie_id)
        movie.update(tmdb_info)
        
        movies.append(movie)
    
    return movies

@app.post("/api/users")
async def create_user(neo4j_session=Depends(get_neo4j)):
    """Creates a new user with incremented ID"""
    try:
        # Get max user ID
        query = """
        MATCH (u:User) 
        WITH COALESCE(MAX(u.userId), 0) as maxId
        CREATE (newUser:User {userId: maxId + 1})
        RETURN newUser.userId as userId
        """
        result = neo4j_session.run(query)
        record = result.single()
        
        if not record:
            raise HTTPException(status_code=500, detail="Failed to create user")
            
        return {"userId": record["userId"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/feedback/recommendation")
async def create_recommendation_feedback(
    user_id: int,
    movie_id: int,
    quality: int,
    neo4j_session = Depends(get_neo4j)
):
    """
    Almacena el feedback sobre la calidad de una recomendación.
    quality: escala del 1 al 5 donde:
    1 = Muy mala
    2 = Mala
    3 = Regular
    4 = Buena
    5 = Excelente
    """
    query = """
    MATCH (u:User {userId: $user_id})
    MATCH (m:Movie {movieId: $movie_id})
    MERGE (u)-[f:RECOMMENDATION_FEEDBACK]->(m)
    SET f.quality = $quality,
        f.timestamp = datetime()
    RETURN f
    """
    try:
        neo4j_session.run(
            query,
            user_id=user_id,
            movie_id=movie_id,
            quality=quality
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/interactions")
async def get_user_interactions(
    user_id: int,
    neo4j_session = Depends(get_neo4j)
):
    query = """
    MATCH (u:User {userId: $user_id})-[r:RATED|GAVE_FEEDBACK]->(m:Movie)
    RETURN m.movieId as movie_id,
           m.title as title,
           type(r) as interaction_type,
           CASE type(r)
             WHEN 'RATED' THEN r.rating
             WHEN 'GAVE_FEEDBACK' THEN r.type
           END as value,
           r.timestamp as timestamp
    ORDER BY r.timestamp DESC
    """
    
    try:
        result = neo4j_session.run(query, user_id=user_id)
        interactions = [dict(record) for record in result]
        return interactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/feedback")
async def create_feedback(
    movie_id: int,
    feedback_type: str,
    user_id: int,
    neo4j_session = Depends(get_neo4j)
):
    query = """
    MATCH (u:User {userId: $user_id})
    MATCH (m:Movie {movieId: $movie_id})
    MERGE (u)-[f:GAVE_FEEDBACK]->(m)
    SET f.type = $feedback_type,
        f.timestamp = datetime()
    RETURN f
    """
    
    try:
        neo4j_session.run(
            query,
            user_id=user_id,
            movie_id=movie_id,
            feedback_type=feedback_type
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/genres")
async def get_genres(neo4j_session = Depends(get_neo4j)):
    query = """
    MATCH (m:Movie)
    UNWIND m.genres as genre
    RETURN DISTINCT genre
    ORDER BY genre
    """
    result = neo4j_session.run(query)
    genres = [record["genre"] for record in result if record["genre"] != "(no genres listed)"]
    return genres

