from py2neo import Graph
import pandas as pd

# Conexión a Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Cargar el archivo genome_tags.csv
tags_csv_path = "data/movielens/genome_tags.csv"
tags_df = pd.read_csv(tags_csv_path)

# Actualizar nodos Tag con tagId
for _, row in tags_df.iterrows():
    tag_id = int(row['tagId'])
    tag_name = row['tag']

    query = """
    MATCH (t:Tag {name: $tag_name})
    SET t.tagId = $tag_id
    """
    graph.run(query, tag_name=tag_name, tag_id=tag_id)

print("Nodos Tag actualizados correctamente.")


