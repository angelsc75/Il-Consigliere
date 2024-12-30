from py2neo import Graph
from dotenv import load_dotenv
import os

# Cargar configuración desde .env (opcional)
load_dotenv()

# Conexión a Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def remove_duplicate_tags():
    # Encuentra los tags duplicados por nombre
    query_find_duplicates = """
    MATCH (t:Tag)
    WITH t.name AS tagName, COLLECT(t) AS tags
    WHERE SIZE(tags) > 1
    RETURN tagName, tags
    """
    duplicates = graph.run(query_find_duplicates).data()

    for duplicate in duplicates:
        tag_name = duplicate["tagName"]
        tags = duplicate["tags"]
        
        print(f"Processing duplicate tag: {tag_name}")

        # Seleccionar el primer nodo como principal
        main_tag = tags[0]

        for duplicate_tag in tags[1:]:
            # Reasignar las relaciones hacia el tag principal
            query_reassign = """
            MATCH (m:Movie)-[r:HAS_TAG]->(duplicate)
            WHERE id(duplicate) = $duplicate_id
            MERGE (m)-[:HAS_TAG]->(main)
            DELETE r
            """
            graph.run(query_reassign, duplicate_id=duplicate_tag.identity, main=main_tag)

            # Eliminar el nodo duplicado
            query_delete = """
            MATCH (duplicate:Tag)
            WHERE id(duplicate) = $duplicate_id
            DELETE duplicate
            """
            graph.run(query_delete, duplicate_id=duplicate_tag.identity)

        print(f"Finished processing {tag_name}")

if __name__ == "__main__":
    try:
        remove_duplicate_tags()
        print("Duplicate tags have been successfully removed.")
    except Exception as e:
        print(f"An error occurred: {e}")
