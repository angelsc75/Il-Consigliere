import pickle
from neo4j import GraphDatabase
import pandas as pd

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=('neo4j', 'neo4jneo4j'))
        
    def close(self):
        self.driver.close()
        
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

def diagnose_movie_ids():
    # 1. Cargar el encoder guardado
    with open('ml/models/model_ok/movie_encoder.pkl', 'rb') as f:
        movie_encoder = pickle.load(f)
    
    # 2. Obtener películas del conjunto de entrenamiento
    trained_movies = set(movie_encoder.classes_)
    print(f"\nEstadísticas del conjunto de entrenamiento:")
    print(f"Número total de películas en training: {len(trained_movies)}")
    print(f"Rango de IDs: {min(trained_movies)} - {max(trained_movies)}")
    
    # 3. Verificar IDs específicos
    problem_ids = [84304, 27876, 48783, 6883]
    print("\nVerificación de IDs problemáticos:")
    for movie_id in problem_ids:
        print(f"ID {movie_id}: {'Presente' if movie_id in trained_movies else 'NO presente'} en training")
    
    # 4. Consultar Neo4j para estos IDs
    conn = Neo4jConnection(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="neo4jneo4j"
    )
    
    movies_query = """
    MATCH (m:Movie)
    WHERE m.movieId IN [84304, 27876, 48783, 6883]
    WITH m
    MATCH (m)<-[r:RATED]-()
    RETURN m.movieId as movieId, 
           m.title as title,
           COUNT(r) as num_ratings
    """
    
    movies_df = conn.query(movies_query)
    print("\nInformación de las películas en Neo4j:")
    print(movies_df)
    
    # 5. Verificar distribución de ratings en el training set
    with open('ml/models/model_ok/ratings_stats.pkl', 'rb') as f:
        ratings_stats = pickle.load(f)
    print("\nEstadísticas de ratings en training:")
    print(f"Media de ratings: {ratings_stats['mean']}")
    print(f"Desviación estándar: {ratings_stats['std']}")
    
    # 6. Verificar estadísticas generales de Neo4j
    stats_query = """
    MATCH (m:Movie)
    OPTIONAL MATCH (m)<-[r:RATED]-()
    RETURN 
        COUNT(DISTINCT m) as total_movies,
        COUNT(DISTINCT r) as total_ratings,
        COUNT(DISTINCT m.movieId) as unique_movie_ids
    """
    
    stats_df = conn.query(stats_query)
    print("\nEstadísticas generales de la base de datos:")
    print(stats_df)
    
    conn.close()

if __name__ == "__main__":
    diagnose_movie_ids()