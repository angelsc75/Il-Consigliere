import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

class MovieLensImporter:
    load_dotenv()
    def __init__(self, uri, user, password):
        """
        Inicializa la conexión con Neo4j
        
        Args:
        - uri: URL de conexión (ej. bolt://localhost:7687)
        - user: usuario de Neo4j
        - password: contraseña de Neo4j
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Cierra la conexión con la base de datos"""
        self.driver.close()

    def import_movies(self, movies_path):
        """
        Importa películas a Neo4j
        
        Args:
        - movies_path: ruta al archivo CSV de películas procesadas
        """
        movies_df = pd.read_csv(movies_path)
        
        with self.driver.session() as session:
            # Crear índices para búsqueda eficiente
            session.run("CREATE INDEX movie_id_index IF NOT EXISTS FOR (m:Movie) ON (m.movieId)")
            
            # Importar películas
            for _, movie in movies_df.iterrows():
                query = """
                MERGE (m:Movie {movieId: $movieId})
                SET 
                    m.title = $title, 
                    m.genres = $genres,
                    m.total_ratings = $total_ratings,
                    m.avg_rating = $avg_rating,
                    m.total_users = $total_users,
                    m.popularity_score = $popularity_score
                """
                
                # Convertir lista de géneros a string
                genres = '|'.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                
                session.run(query, {
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': genres,
                    'total_ratings': movie.get('total_ratings', 0),
                    'avg_rating': movie.get('avg_rating', 0),
                    'total_users': movie.get('total_users', 0),
                    'popularity_score': movie.get('popularity_score', 0)
                })

    def import_ratings(self, ratings_path):
        """
        Importa ratings y crea relaciones entre usuarios y películas
        
        Args:
        - ratings_path: ruta al archivo CSV de ratings procesados
        """
        ratings_df = pd.read_csv(ratings_path)
        
        with self.driver.session() as session:
            # Crear índices
            session.run("CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.userId)")
            
            # Importar ratings
            for _, rating in ratings_df.iterrows():
                query = """
                MERGE (u:User {userId: $userId})
                MERGE (m:Movie {movieId: $movieId})
                MERGE (u)-[r:RATED]->(m)
                SET 
                    r.rating = $rating, 
                    r.timestamp = $timestamp
                """
                
                session.run(query, {
                    'userId': rating['userId'],
                    'movieId': rating['movieId'],
                    'rating': rating['rating'],
                    'timestamp': rating['timestamp']
                })

def main():
    # Configuración de conexión
    URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    USER =  os.getenv('NEO4J_USER', 'neo4j')
    PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    
    # Rutas de archivos procesados
    DATASET_PATH = 'data/movielens/processed'
    MOVIES_PATH = os.path.join(DATASET_PATH, 'movies_processed.csv')
    RATINGS_PATH = os.path.join(DATASET_PATH, 'ratings_processed.csv')
    
    # Inicializar importador
    importer = MovieLensImporter(URI, USER, PASSWORD)
    
    try:
        # Importar datos
        importer.import_movies(MOVIES_PATH)
        importer.import_ratings(RATINGS_PATH)
        print("Importación completada exitosamente!")
    
    finally:
        # Cerrar conexión
        importer.close()

if __name__ == '__main__':
    main()