from py2neo import Graph
import pandas as pd

# Conexión a Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Cargar el archivo genome_scores.csv
genome_scores_csv_path = "data/movielens/genome_scores.csv"
genome_scores_df = pd.read_csv(genome_scores_csv_path)

# Iterar sobre las filas para actualizar relevancia
for _, row in genome_scores_df.iterrows():
    movie_id = int(row['movieId'])
    tag_id = int(row['tagId'])
    relevance = float(row['relevance'])

    # Actualizar relevancia en las relaciones
    query = """
    MATCH (m:Movie {movieId: $movie_id})-[r:HAS_TAG]->(t:Tag {tagId: $tag_id})
    SET r.relevance = $relevance
    """
    graph.run(query, movie_id=movie_id, tag_id=tag_id, relevance=relevance)

print("Relevancia añadida a las relaciones HAS_TAG correctamente.")


