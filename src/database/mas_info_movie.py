from py2neo import Graph, Node, NodeMatcher
import pandas as pd
import re

# Conexión a la base de datos Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4jneo4j"))

# Cargar el archivo movie.csv con pandas
movies_df = pd.read_csv("data/movielens/movie.csv")

# Función para limpiar el título (eliminar el año en paréntesis)
def clean_title(title):
    return re.sub(r'\s\(\d{4}\)$', '', title).strip()  # Quita "(YYYY)" al final

# Iterar sobre cada película en el archivo
matcher = NodeMatcher(graph)  # Para buscar nodos existentes

for _, row in movies_df.iterrows():
    movie_id = int(row['movieId'])
    title = clean_title(row['title'])  # Limpiar el título
    genres = row['genres'].split('|') if isinstance(row['genres'], str) else []  # Convertir géneros a lista
    
    # Buscar si el nodo Movie ya existe
    movie_node = matcher.match("Movie", movieId=movie_id).first()
    
    if movie_node:
        # Si el nodo existe, actualiza los campos faltantes
        movie_node["title"] = title
        movie_node["genres"] = genres
        graph.push(movie_node)  # Guardar los cambios
    else:
        # Si el nodo no existe, créalo
        new_movie = Node("Movie", movieId=movie_id, title=title, genres=genres)
        graph.create(new_movie)  # Guardar el nuevo nodo
