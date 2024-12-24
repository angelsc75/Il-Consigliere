from tensorflow.keras.models import load_model
import pickle
import numpy as np

def load_prediction_resources():
    try:
        # Cargar el modelo
        model = load_model('src/models/model_ok/movie_recommender_model.keras')
        
        # Cargar encoders y features
        with open('src/models/model_ok/user_encoder.pkl', 'rb') as f:
            user_encoder = pickle.load(f)
        with open('src/models/model_ok/movie_encoder.pkl', 'rb') as f:
            movie_encoder = pickle.load(f)
        with open('src/models/model_ok/tag_scaler.pkl', 'rb') as f:
            tag_scaler = pickle.load(f)
        with open('src/models/model_ok/tag_columns.pkl', 'rb') as f:
            tag_columns = pickle.load(f)
        with open('src/models/model_ok/tag_features.pkl', 'rb') as f:
            tag_features = pickle.load(f)
            
        return model, user_encoder, movie_encoder, tag_features
    except Exception as e:
        print(f"Error cargando los recursos: {str(e)}")
        return None

def predict_rating(model, user_id, movie_id, user_encoder, movie_encoder, tag_features):
    try:
        # Cargar estadísticas de normalización
        with open('src/models/model_ok/ratings_stats.pkl', 'rb') as f:
            ratings_stats = pickle.load(f)
        
        # Codificar inputs
        user_encoded = user_encoder.transform([user_id])
        movie_encoded = movie_encoder.transform([movie_id])
        movie_tags = tag_features[movie_encoded[0]]
        
        # Obtener predicción normalizada
        prediction_normalized = model.predict(
            [user_encoded, movie_encoded, movie_tags.reshape(1, -1)],
            verbose=0
        )
        
        # Desnormalizar la predicción
        prediction = (prediction_normalized[0][0] * ratings_stats['std']) + ratings_stats['mean']
        
        # Asegurar que la predicción está en el rango válido (1-5)
        prediction = max(1, min(5, prediction))
        
        return prediction
    except Exception as e:
        print(f"Error en predict_rating: {str(e)}")
        return None

def predict_for_user(user_id, movie_id):
    # Cargar recursos
    resources = load_prediction_resources()
    if resources is None:
        return
    
    model, user_encoder, movie_encoder, tag_features = resources
    
    # Hacer predicción
    try:
        rating = predict_rating(model, user_id, movie_id, user_encoder, movie_encoder, tag_features)
        if rating is not None:
            print(f"Predicción para usuario {user_id}, película {movie_id}: {rating:.2f} estrellas")
    except Exception as e:
        print(f"Error al hacer la predicción: {str(e)}")

def recommend_movies_for_user(user_id, n_recommendations=5):
    # Cargar recursos
    resources = load_prediction_resources()
    if resources is None:
        return
    
    model, user_encoder, movie_encoder, tag_features = resources
    
    predictions = []
    for movie_id in movie_encoder.classes_:
        try:
            rating = predict_rating(model, user_id, movie_id, user_encoder, movie_encoder, tag_features)
            if rating is not None:
                predictions.append((movie_id, rating))
        except Exception as e:
            continue
    
    # Ordenar por rating predicho
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {n_recommendations} recomendaciones para usuario {user_id}:")
    for movie_id, rating in predictions[:n_recommendations]:
        print(f"Película {movie_id}: {rating:.2f} estrellas")

if __name__ == "__main__":
    # Ejemplo de uso
    user_id = 1  # Reemplaza con un ID de usuario real
    movie_id = 100  # Reemplaza con un ID de película real
    
    # Probar una predicción individual
    predict_for_user(user_id, movie_id)
    
    # Probar recomendaciones
    recommend_movies_for_user(user_id, n_recommendations=5)