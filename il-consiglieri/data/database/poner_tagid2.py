from py2neo import Graph
import pandas as pd

# Conexión a la base de datos Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Cargar el archivo combinado
tags_csv_path = "data/movielens/processed/combined_tags.csv"
tags_df = pd.read_csv(tags_csv_path)

# Actualizar o crear nodos Tag
for _, row in tags_df.iterrows():
    tag_id = int(row['tagId'])
    tag_name = row['tag']

    query = """
    MERGE (t:Tag {name: $tag_name})
    SET t.tagId = $tag_id
    """
    graph.run(query, tag_name=tag_name, tag_id=tag_id)

print("Nodos Tag cargados correctamente.")

