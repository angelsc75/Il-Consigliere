import pandas as pd
import numpy as np
import os

def prepare_movielens_dataset(dataset_path):
    """
    Preparar y limpiar dataset de MovieLens
    
    Pasos:
    1. Cargar archivos
    2. Limpiar datos
    3. Enriquecer dataset
    4. Guardar versión procesada
    """
    
    # Cargar archivos
    movies = pd.read_csv(os.path.join(dataset_path, 'movie.csv'))
    ratings = pd.read_csv(os.path.join(dataset_path, 'rating.csv'))
    
    # Limpiar nombres de películas
    movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    
    # Separar géneros
    movies['genres'] = movies['genres'].str.split('|')
    
    # Estadísticas de ratings
    ratings_stats = ratings.groupby('movieId').agg({
        'rating': ['count', 'mean'],
        'userId': 'nunique'
    }).reset_index()
    
    ratings_stats.columns = ['movieId', 'total_ratings', 'avg_rating', 'total_users']
    
    # Merge con movies
    movies_enriched = movies.merge(ratings_stats, on='movieId', how='left')
    
    # Filtrar películas con suficientes ratings
    movies_enriched = movies_enriched[movies_enriched['total_ratings'] >= 10]
    
    # Añadir columna de popularidad
    movies_enriched['popularity_score'] = (
        movies_enriched['total_ratings'] * 
        movies_enriched['avg_rating'] / 
        movies_enriched['total_ratings'].max()
    )
    
    # Preparar datos de usuarios
    user_activity = ratings.groupby('userId').agg({
        'movieId': 'count',
        'rating': 'mean'
    }).reset_index()
    
    user_activity.columns = ['userId', 'total_movies_rated', 'avg_user_rating']
    
    # Guardar datasets procesados
    output_path = os.path.join(dataset_path, 'processed')
    os.makedirs(output_path, exist_ok=True)
    
    # movies_enriched.to_csv(os.path.join(output_path, 'movies_processed.csv'), index=False)
    # ratings.to_csv(os.path.join(output_path, 'ratings_processed.csv'), index=False)
    # user_activity.to_csv(os.path.join(output_path, 'user_activity.csv'), index=False)
    
    # Resumen del procesamiento
    print("\n--- Resumen de Procesamiento de Dataset ---")
    print(f"Películas originales: {len(movies)}")
    print(f"Películas después de filtrar: {len(movies_enriched)}")
    print(f"Total de ratings: {len(ratings)}")
    print(f"Número de usuarios únicos: {ratings['userId'].nunique()}")
    
    return movies_enriched, ratings, user_activity

def split_train_test(ratings, test_size=0.2):
    """
    Dividir datos en conjunto de entrenamiento y prueba
    """
    # Stratified split por usuario
    users = ratings['userId'].unique()
    np.random.shuffle(users)
    
    split_index = int(len(users) * (1 - test_size))
    train_users = users[:split_index]
    test_users = users[split_index:]
    
    train_ratings = ratings[ratings['userId'].isin(train_users)]
    test_ratings = ratings[ratings['userId'].isin(test_users)]
    
    return train_ratings, test_ratings

def main():
    # Ruta al dataset descargado
    dataset_path = 'data/movielens'
    
    # Procesar dataset
    movies, ratings, user_activity = prepare_movielens_dataset(dataset_path)
    
    # Dividir en train y test
    train_ratings, test_ratings = split_train_test(ratings)
    
    # Guardar splits
    # train_ratings.to_csv(os.path.join(dataset_path, 'processed', 'train_ratings.csv'), index=False)
    # test_ratings.to_csv(os.path.join(dataset_path, 'processed', 'test_ratings.csv'), index=False)

if __name__ == '__main__':
    main()