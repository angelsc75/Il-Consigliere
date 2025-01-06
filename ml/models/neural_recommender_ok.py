from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=('neo4j', 'neo4jneo4j'))
        
    def close(self):
        self.driver.close()
        
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_neo4j(conn, limit=1000000, batch_size=5000):
    # Obtener el total de películas y usuarios
    logging.info("Obteniendo conteo total de películas y usuarios desde Neo4j...")
    movie_count_query = "MATCH (m:Movie) RETURN count(m) as total_movies"
    user_count_query = "MATCH (u:User) RETURN count(u) as total_users"

    total_movies = conn.query(movie_count_query)['total_movies'][0]
    total_users = conn.query(user_count_query)['total_users'][0]

    logging.info(f"Total de películas: {total_movies}, Total de usuarios: {total_users}")

    # Consultar ratings balanceados por película
    logging.info("Cargando datos de ratings balanceados por película...")
    ratings_query = f"""
    MATCH (u:User)-[r:RATED]->(m:Movie)
    WITH m, COLLECT([u.userId, r.rating]) AS ratings
    WITH m.movieId AS movieId, m.title AS title, ratings[0..{limit // total_movies}] AS sampled_ratings
    UNWIND sampled_ratings AS r
    RETURN r[0] AS userId, movieId, r[1] AS rating
    """
    ratings_df = conn.query(ratings_query)
    logging.info(f"Cargados {len(ratings_df)} ratings en total.")

    # Cargar tags y sus relevancias
    logging.info("Cargando tags de películas desde Neo4j...")
    tags_query = """
    MATCH (m:Movie)-[t:HAS_TAG]->(tag:Tag)
    RETURN m.movieId as movieId, tag.name as tagName, t.relevance as relevance
    """
    tags_df = conn.query(tags_query)

    logging.info(f"Se cargaron {len(tags_df)} filas de tags.")

    return ratings_df, tags_df

# Modificar prepare_tag_features para garantizar todas las películas

def prepare_tag_features(tags_df, all_movie_ids, movie_encoder):
    logging.info("Preparando características de tags para todas las películas...")
    # Crear matriz de tags con pivoteo
    tag_matrix = tags_df.pivot_table(
        index='movieId',
        columns='tagName',
        values='relevance',
        fill_value=0
    )

    # Asegurar que todas las películas estén incluidas
    missing_movies = set(movie_encoder.classes_) - set(tag_matrix.index)
    if missing_movies:
        logging.info(f"Agregando {len(missing_movies)} películas faltantes sin tags.")
        missing_df = pd.DataFrame(0, 
                                  index=list(missing_movies),
                                  columns=tag_matrix.columns)
        tag_matrix = pd.concat([tag_matrix, missing_df])

    # Reordenar matriz según el orden de movie_encoder
    tag_matrix = tag_matrix.reindex(movie_encoder.classes_, fill_value=0)

    # Normalizar características de tags
    logging.info("Normalizando características de tags...")
    scaler = StandardScaler()
    tag_features = scaler.fit_transform(tag_matrix)

    # Guardar scaler y nombres de tags para uso futuro
    
    logging.info("Guardando scaler y columnas de tags...")
    with open('ml/models/model_ok/tag_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('ml/models/model_ok/tag_columns.pkl', 'wb') as f:
        pickle.dump(list(tag_matrix.columns), f)

    return tag_features, tag_matrix.columns

# Modificar prepare_data para manejar correctamente el límite y diversidad

def prepare_data(ratings_df, tags_df):
    logging.info("Preparando datos de entrenamiento...")
    # Obtener todas las películas únicas
    all_movie_ids = ratings_df['movieId'].unique()

    # Verificar cuántas películas únicas están en los datos cargados
    logging.info(f"Número total de películas únicas en ratings_df: {len(all_movie_ids)}")

    # Codificar usuarios y películas
    logging.info("Codificando usuarios y películas...")
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    # Incluir todas las películas únicas para garantizar codificación completa
    movie_encoder.fit(pd.concat([pd.Series(all_movie_ids), tags_df['movieId']]).unique())
    ratings_df['user_encoded'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movie_encoded'] = movie_encoder.transform(ratings_df['movieId'])

    # Verificar cuántas películas están codificadas
    logging.info(f"Películas codificadas en movie_encoder: {len(movie_encoder.classes_)}")

    # Normalizar ratings
    logging.info("Normalizando ratings...")
    ratings_mean = ratings_df['rating'].mean()
    ratings_std = ratings_df['rating'].std()
    ratings_df['rating'] = (ratings_df['rating'] - ratings_mean) / ratings_std

    # Guardar estadísticas de normalización
    
    logging.info("Guardando estadísticas de normalización...")
    with open('ml/models/model_ok/ratings_stats.pkl', 'wb') as f:
        pickle.dump({'mean': ratings_mean, 'std': ratings_std}, f)

    # Preparar características de tags
    tag_features, tag_columns = prepare_tag_features(
        tags_df, 
        all_movie_ids,
        movie_encoder
    )

    # Guardar tag_features para uso en predicciones
    logging.info("Guardando características de tags...")
    with open('ml/models/model_ok/tag_features.pkl', 'wb') as f:
        pickle.dump(tag_features, f)

    # Guardar encoders
    logging.info("Guardando encoders de usuarios y películas...")
    with open('ml/models/model_ok/user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open('ml/models/model_ok/movie_encoder.pkl', 'wb') as f:
        pickle.dump(movie_encoder, f)

    # Registrar la cantidad de películas finales procesadas
    logging.info(f"Total de películas finales procesadas: {len(movie_encoder.classes_)}")

    return ratings_df, tag_features, user_encoder, movie_encoder, len(tag_columns)

def create_model(num_users, num_movies, num_tag_features, embedding_size=50):
    # Input layers
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')
    tag_input = layers.Input(shape=(num_tag_features,), name='tag_input')
    
    # Embedding layers
    user_embedding = layers.Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    movie_embedding = layers.Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)
    
    # Flatten embeddings
    user_flatten = layers.Flatten()(user_embedding)
    movie_flatten = layers.Flatten()(movie_embedding)
    
    # Process tag features
    tag_dense = layers.Dense(32, activation='relu')(tag_input)
    
    # Concatenate all features
    concat = layers.Concatenate()([user_flatten, movie_flatten, tag_dense])
    
    # Dense layers
    dense1 = layers.Dense(96, activation='relu')(concat)
    dropout1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(48, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    dense3 = layers.Dense(24, activation='relu')(dropout2)
    
    # Output layer
    output = layers.Dense(1)(dense3)
    
    # Create model
    model = models.Model(
        inputs=[user_input, movie_input, tag_input],
        outputs=output
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def prepare_model_inputs(ratings_df, movie_tag_features):
    X_users = ratings_df['user_encoded'].values
    X_movies = ratings_df['movie_encoded'].values
    X_tags = np.array([movie_tag_features[mid] for mid in X_movies])
    y = ratings_df['rating'].values
    
    return X_users, X_movies, X_tags, y

def train_and_evaluate(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
    # Añadir early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Añadir learning rate scheduling
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2
    )
    
    history = model.fit(
        [X_train[0], X_train[1], X_train[2]], y_train,
        validation_data=([X_val[0], X_val[1], X_val[2]], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Conectar a Neo4j
    conn = Neo4jConnection(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="neo4jneo4j"
    )
    
    # Cargar datos
    ratings_df, tags_df = load_data_from_neo4j(conn, limit=1000000)
    
    # Preparar datos
    ratings_df, movie_tag_features, user_encoder, movie_encoder, num_tag_features = prepare_data(ratings_df, tags_df)
    
    # Preparar inputs del modelo
    X_users, X_movies, X_tags, y = prepare_model_inputs(ratings_df, movie_tag_features)
    
    # Split de datos
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = [X_users[train_idx], X_movies[train_idx], X_tags[train_idx]]
    X_val = [X_users[val_idx], X_movies[val_idx], X_tags[val_idx]]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Crear y entrenar modelo
    model = create_model(
        num_users=len(user_encoder.classes_),
        num_movies=len(movie_encoder.classes_),
        num_tag_features=num_tag_features
    )
    
    # Entrenar modelo
    history = train_and_evaluate(model, X_train, y_train, X_val, y_val)
    
    # Visualizar resultados
    plot_training_history(history)
    
    # Guardar modelo
    model.save('ml/models/model_ok/movie_recommender_model.keras')
    
    # Cerrar conexión
    conn.close()

def predict_rating(model, user_id, movie_id, user_encoder, movie_encoder, tag_features):
    # Cargar estadísticas de normalización
    with open('ml/models/model_ok/ratings_stats.pkl', 'rb') as f:
        ratings_stats = pickle.load(f)
    
    # Codificar inputs
    user_encoded = user_encoder.transform([user_id])
    movie_encoded = movie_encoder.transform([movie_id])
    movie_tags = tag_features[movie_encoder.transform([movie_id])[0]]
    
    # Obtener predicción normalizada
    prediction_normalized = model.predict([
        user_encoded,
        movie_encoded,
        movie_tags.reshape(1, -1)
    ])
    
    # Desnormalizar la predicción
    prediction = (prediction_normalized[0][0] * ratings_stats['std']) + ratings_stats['mean']
    
    # Asegurar que la predicción está en el rango válido (ejemplo: 1-5 para ratings de películas)
    prediction = max(1, min(5, prediction))
    
    return prediction

if __name__ == "__main__":
    main()