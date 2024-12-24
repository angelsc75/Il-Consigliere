from py2neo import Graph, NodeMatcher
import pandas as pd

# Conexión a la base de datos Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Campos a eliminar
fields_to_remove = ["popularity_score", "total_ratings", "total_users", "avg_rating"]

# Usar NodeMatcher para encontrar nodos Movie
matcher = NodeMatcher(graph)
all_movies = matcher.match("Movie")  # Selecciona todos los nodos Movie

# Iterar sobre cada nodo Movie y eliminar los campos
for movie_node in all_movies:
    for field in fields_to_remove:
        if field in movie_node:
            movie_node.pop(field)  # Eliminar el campo del nodo
    graph.push(movie_node)  # Guardar los cambios en la base de datos

print("Campos eliminados correctamente.")