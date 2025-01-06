from py2neo import Graph
import pandas as pd

# Conexión a la base de datos Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Ruta al archivo CSV
csv_path = "data/movielens/genome_scores.csv"

# Cargar el archivo CSV y limpiar los datos
genome_scores_df = pd.read_csv(csv_path)
genome_scores_df = genome_scores_df.dropna(subset=['movieId', 'tagId', 'relevance'])
genome_scores_df['movieId'] = genome_scores_df['movieId'].astype(int)
genome_scores_df['tagId'] = genome_scores_df['tagId'].astype(int)
genome_scores_df['relevance'] = genome_scores_df['relevance'].astype(float)

# Procesar en lotes
batch_size = 1000
for i in range(0, len(genome_scores_df), batch_size):
    batch_df = genome_scores_df.iloc[i:i + batch_size]
    batch = batch_df.to_dict(orient='records')
    print(f"Procesando lote {i // batch_size + 1} con {len(batch)} filas.")

    try:
        query = """
        UNWIND $batch AS row
        MATCH (m:Movie {movieId: row.movieId}), (t:Tag {tagId: row.tagId})
        MERGE (m)-[r:HAS_TAG]->(t)
        SET r.relevance = row.relevance
        """
        graph.run(query, batch=batch)
    except Exception as e:
        print(f"Error al procesar el lote {i // batch_size + 1}: {e}")

print("Trabajo finalizado.")


print("Relaciones HAS_TAG creadas correctamente.")

