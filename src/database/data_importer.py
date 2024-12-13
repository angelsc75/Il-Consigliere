import pandas as pd
from .neo4j_connection import Neo4jConnection
import os

class MovieLensImporter:
    def __init__(self, neo4j_conn: Neo4jConnection):
        """
        Inicializador del importador de MovieLens
        
        :param neo4j_conn: Conexión a Neo4j previamente establecida
        """
        self.neo4j_conn = neo4j_conn
        self.data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'movielens')

    def load_data(self, filename):
        """
        Cargar archivo CSV
        
        :param filename: Nombre del archivo a cargar
        :return: DataFrame con los datos
        """
        filepath = os.path.join(self.data_path, filename)
        return pd.read_csv(filepath)

    def import_movies(self):
        """Importar nodos de películas"""
        movies_df = self.load_data('movies.csv')
        
        query = """
        UNWIND $movies AS movie
        MERGE (m:Movie {movieId: movie.movieId})
        SET m.title = movie.title,
            m.genres = movie.genres
        """
        
        with self.neo4j_conn.get_session() as session:
            session.run(query, movies=movies_df.to_dict('records'))
        
        print(f"Importados {len(movies_df)} películas")

    def import_ratings(self):
        """Importar relaciones de ratings"""
        ratings_df = self.load_data('ratings.csv')
        
        query = """
        UNWIND $ratings AS rating
        MATCH (m:Movie {movieId: rating.movieId})
        MERGE (u:User {userId: rating.userId})
        MERGE (u)-[r:RATED]->(m)
        SET r.rating = rating.rating,
            r.timestamp = rating.timestamp
        """
        
        with self.neo4j_conn.get_session() as session:
            session.run(query, ratings=ratings_df.to_dict('records'))
        
        print(f"Importados {len(ratings_df)} ratings")

    def import_tags(self):
        """Importar tags de películas"""
        tags_df = self.load_data('tags.csv')
        
        query = """
        UNWIND $tags AS tag
        MATCH (m:Movie {movieId: tag.movieId})
        MATCH (u:User {userId: tag.userId})
        MERGE (u)-[:TAGGED {timestamp: tag.timestamp}]->(m)
        SET m.tags = coalesce(m.tags, []) + tag.tag
        """
        
        with self.neo4j_conn.get_session() as session:
            session.run(query, tags=tags_df.to_dict('records'))
        
        print(f"Importados {len(tags_df)} tags")

    def import_all(self):
        """Importar todos los datos y crear constrains"""
        # Crear constrains primero
        self.neo4j_conn.create_constraints()
        
        # Importar datos
        self.import_movies()
        self.import_ratings()
        self.import_tags()

# Ejemplo de uso
def main():
    # Crear conexión a Neo4j
    neo4j_conn = Neo4jConnection()
    
    try:
        # Crear importador
        importer = MovieLensImporter(neo4j_conn)
        
        # Importar todos los datos
        importer.import_all()
    
    except Exception as e:
        print(f"Error durante la importación: {e}")
    
    finally:
        # Cerrar conexión
        neo4j_conn.close()

if __name__ == "__main__":
    main()