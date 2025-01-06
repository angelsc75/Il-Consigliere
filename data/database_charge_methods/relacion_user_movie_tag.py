from py2neo import Graph
import pandas as pd

# Conexión a Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Cargar el archivo tag.csv
tags_csv_path = "data/movielens/processed/tag_ok.csv"
tags_df = pd.read_csv(tags_csv_path)

# Iterar sobre las filas para crear las relaciones
for _, row in tags_df.iterrows():
    user_id = int(row['userId'])
    movie_id = int(row['movieId'])
    tag_name = row['tag']
    timestamp = row['timestamp']  # Timestamp del tag

    # Relacionar User, Movie y Tag con timestamp
    query = """
    MATCH (t:Tag {name: $tag_name})
    MERGE (u:User {userId: $user_id})
    MERGE (m:Movie {movieId: $movie_id})
    MERGE (u)-[r:TAGGED]->(m)
    SET r.tag = $tag_name, r.timestamp = $timestamp
    MERGE (m)-[:HAS_TAG]->(t)
    """
    graph.run(query, user_id=user_id, movie_id=movie_id, tag_name=tag_name, timestamp=timestamp)

print("Relaciones TAGGED y HAS_TAG creadas correctamente con timestamp.")
