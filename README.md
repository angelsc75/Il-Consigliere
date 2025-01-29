# ![Il Consigliere Logo](images/logo.png) Il consigliere (Un Recomendador de Películas)

## Descripción
Nota: este proyecto forma parte de los realizados en el bootcamp de IA de Factoría F5 realizado de
mayo de 2024 a marzo de 2025. Es un proyecto personal, en el que yo elegí el tema ya que quería investigar cómo hacer un modelo recomendador y usar una base de datos de grafos.

Este proyecto es una aplicación completa que permite a los usuarios obtener recomendaciones personalizadas de películas basadas en sus puntuaciones previas. Utiliza **FastAPI** para el backend, **React** para el frontend y una base de datos **Neo4j** para almacenar y gestionar los datos. Un modelo de red neuronal se entrena para predecir las calificaciones de los usuarios y generar recomendaciones personalizadas.

El dataset es MovieLens 20M

## Características

- **Registro y autenticación de usuarios**: Los usuarios pueden crear cuentas y calificar películas.
- **Calificaciones**: Los usuarios pueden puntuar películas del catálogo.
- **Recomendaciones personalizadas**: Basadas en las calificaciones y el modelo entrenado.
- **Búsqueda de películas**: Por título, etiquetas y géneros.
- **Feedback de recomendaciones**: Los usuarios pueden valorar la calidad de las recomendaciones para mejorar el modelo (esta parte se queda para las mejoras, en las que se pretende reforzar el aprendizaje del modelo...)

---

## Estructura del Proyecto

### Frontend
- **`OnboardingFlow.js`**  
  Este componente guía a los usuarios desde la selección de su cuenta hasta las recomendaciones personalizadas. Incluye las siguientes características:
  - Crear cuentas nuevas o usar un ID existente.
  - Puntuar películas populares.
  - Buscar películas por título, etiqueta o género.
  - Navegar entre las recomendaciones y las búsquedas.

- **`RecommendationRatingScreen.js`**  
  Una interfaz para que los usuarios evalúen recomendaciones y califiquen películas con estrellas. También permite valorar la calidad de las recomendaciones.

### Backend
- **`api.py`**  
  Proporciona los endpoints de la API para interactuar con el frontend y gestionar las operaciones principales, incluyendo:
  - **Gestión de usuarios**: Crear usuarios, verificar su estado, obtener interacciones.
  - **Gestión de películas**: Buscar películas, obtener populares y generar recomendaciones personalizadas.
  - **Recomendaciones**: Integra un modelo de red neuronal para predecir calificaciones.
  - **Feedback**: Almacenar las opiniones sobre las recomendaciones.

### Modelo de Recomendación
- Entrenado utilizando datos de **MovieLens**.
- Utiliza un **modelo de red neuronal** con entradas como:
  - IDs de usuario y película codificados.
  - Características de las películas basadas en etiquetas.
- Utiliza **Keras/TensorFlow** para predecir calificaciones.

---

## Instalación

### Requisitos
- Python 3.8+
- Node.js 16+
- Neo4j
- Dependencias adicionales: listadas en `requirements.txt` y `package.json`.

### Backend
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-repositorio.git
   cd tu-repositorio
   ```
2. Crea un entorno virtual y activa:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate  # Windows
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configura las variables de entorno en un archivo `.env`:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=tu_usuario
   NEO4J_PASSWORD=tu_contraseña
   TMDB_API_KEY=tu_api_key
   ```
5. Ejecuta el backend:
Antes de ejecutar el backend hay que ejecutar el archivo neural_recommender_ok.py, que creará a su vez otros archivos que necesitará api.py para funcionar correctamente. Los archivos generados por el modelo es posible que no estén actualizados porque uno de los actualizados pesaba demasiado como para poder subirlo al repositorio de GitHub. Una vez ejecutado este archivo. Ya puedes arrancar el backend.

   ```bash
   uvicorn api:app --reload
   ```

### Frontend
1. Ve a la carpeta del frontend:
   ```bash
   cd frontend
   ```
2. Instala las dependencias:
   ```bash
   npm install
   ```
3. Ejecuta el servidor de desarrollo:
   ```bash
   npm start
   ```

---

## Uso

1. Accede a la aplicación desde el navegador:  
   **Frontend**: [http://localhost:3000](http://localhost:3000)  
   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

2. **Navega por las funcionalidades principales**:
   - Crea un usuario y califica películas.
   - Explora películas populares o busca por título, género o etiqueta.
   - Obtén recomendaciones personalizadas basadas en tus puntuaciones.

---

## Archivos Clave

### Frontend
- **`OnboardingFlow.js`**: Flujo principal de navegación del usuario.
- **`RecommendationRatingScreen.js`**: Interfaz para calificar recomendaciones.

### Backend
- **`api.py`**: API REST que conecta el frontend con los datos y el modelo de recomendación.

---

## Próximos Pasos

1. Mejorar la interfaz gráfica para un flujo de usuario más fluido.
2. Optimizar el modelo de recomendación para manejar grandes volúmenes de datos.
3. Añadir autenticación avanzada con OAuth2.
4. Incluir pruebas unitarias y de integración.
5. Mejorar la gestión de usuarios no incluidos en el entrenamiento del modelo.

---

