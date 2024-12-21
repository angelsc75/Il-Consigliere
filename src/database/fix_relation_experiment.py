from py2neo import Graph, NodeMatcher
import pandas as pd

# Conexión a la base de datos Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Ruta al archivo CSV en tu proyecto
csv_path = "data/movielens/genome_scores.csv"

# Cargar los datos del CSV usando pandas
genome_scores_df = pd.read_csv(csv_path)

# Filtrar las filas solo para movieId=1
subset_df = genome_scores_df[genome_scores_df['movieId'] == 1]

# Iterar sobre el subconjunto
for _, row in subset_df.iterrows():
    movie_id = int(row['movieId'])
    tag_id = int(row['tagId'])
    relevance = float(row['relevance'])

    # Buscar nodos existentes
    matcher = NodeMatcher(graph)
    movie_node = matcher.match("Movie", movieId=movie_id).first()
    tag_node = matcher.match("Tag", tagId=tag_id).first()
    
    if movie_node and tag_node:
        # Crear o actualizar la relación HAS_TAG
        query = """
        MATCH (m:Movie {movieId: $movie_id}), (t:Tag {tagId: $tag_id})
        MERGE (m)-[r:HAS_TAG]->(t)
        SET r.relevance = $relevance
        """
        graph.run(query, movie_id=movie_id, tag_id=tag_id, relevance=relevance)

print("Relaciones para movieId=1 creadas correctamente.")