import React, { useState, useEffect } from 'react';
import Rating from 'react-rating'; // Componente para estrellas
import { Search, Star, ThumbsUp, ThumbsDown, Loader, Film } from 'lucide-react';
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent, 
  CardFooter 
} from './components/ui/card';
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import { Progress } from "./components/ui/progress";
import Header from "./components/ui/header";
import RecommendationRatingScreen from './RecommendationRatingScreen';

const API_BASE_URL = 'http://localhost:8000';



const MovieCard = ({ 
  movie, 
  onRate, 
  currentRating, 
  showFeedback = false, 
  onFeedback,
  setSearchQuery,
  setSearchType,
  handleSearch,
  setActiveTab 
}) => {
  if (!movie) return null;

  return (
    <Card className="w-full max-w-sm bg-card text-card-foreground shadow-sm hover:shadow-lg transition-shadow group">
      <div className="relative">
        {movie.poster_path ? (
          <img
            src={movie.poster_path}
            alt={movie.title}
            className="w-full aspect-[2/3] object-cover"
          />
        ) : (
          <div className="w-full aspect-[2/3] bg-gray-200 flex items-center justify-center">
            <Film className="w-12 h-12 text-gray-400" />
          </div>
        )}
        <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
          {movie.rating ? `Rating: ${movie.rating}` : 'No Rating'}
        </div>
      </div>
      <CardContent className="p-4">
        <CardTitle className="text-lg font-semibold mb-2 text-gray-800">{movie.title}</CardTitle>
        {movie.overview && (
          <CardDescription
            className="text-sm text-gray-600 line-clamp-10 group-hover:line-clamp-none transition-all duration-300 ease-in-out"
          >
            {movie.overview}
          </CardDescription>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          {movie.tags
            ?.sort((a, b) => b.relevance - a.relevance)
            .slice(0, 10)
            .map((tag, index) => (
              <button
                key={`${tag.name}-${index}`}
                onClick={() => {
                  setSearchQuery(tag.name || tag);
                  setSearchType('tag');
                  handleSearch();
                  setActiveTab('search');
                }}
                className="px-2 py-1 bg-blue-100 text-blue-600 rounded-full text-xs 
                        hover:bg-blue-200 transition-colors cursor-pointer"
              >
                {tag.name || tag}
              </button>
            ))}
        </div>
      </CardContent>
      <CardFooter className="p-4 flex justify-between items-center border-t">
        {onRate && (
          <div className="flex items-center gap-2">
            <Rating
              initialRating={currentRating || 0}
              fractions={2} // Permite media estrella
              emptySymbol={<Star className="text-gray-300 w-6 h-6" />}
              fullSymbol={<Star className="text-yellow-500 w-6 h-6" />}
              onChange={(value) => onRate(movie.id, value)}
            />
            <span className="text-sm text-gray-600">{currentRating || 'Sin calificar'}</span>
          </div>
        )}
        {showFeedback && (
          <div className="flex gap-2">
            <Button
              onClick={() => onFeedback(movie.id, 'like')}
              variant="outline"
              className="flex gap-2 text-green-500 border-green-500 hover:bg-green-100"
            >
              <ThumbsUp className="w-4 h-4" />
            </Button>
            <Button
              onClick={() => onFeedback(movie.id, 'dislike')}
              variant="outline"
              className="flex gap-2 text-red-500 border-red-500 hover:bg-red-100"
            >
              <ThumbsDown className="w-4 h-4" />
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};

const OnboardingFlow = () => {
  const [userId, setUserId] = useState(null);
  const [inputUserId, setInputUserId] = useState('');
  const [hasEnoughRatings, setHasEnoughRatings] = useState(false);
  const [step, setStep] = useState('user-selection');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState('title');
  const [moviesToRate, setMoviesToRate] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [userRatings, setUserRatings] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState('search');
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [currentMovieRating, setCurrentMovieRating] = useState(null);
  const [recommendationQuality, setRecommendationQuality] = useState(null);
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [minRequiredRatings] = useState(5);
  useEffect(() => {
    const storedUserId = localStorage.getItem('userId');
    if (storedUserId) {
      checkUserStatus(storedUserId);
    }
  }, []);

  const createUser = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Error creating user');
      }
      
      const data = await response.json();
      await checkUserStatus(data.userId);
    } catch (error) {
      console.error('Error creating user:', error);
      setError('Error al crear usuario');
    }
  };
  const checkModelStatus = async (userId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/${userId}/status`);
      const data = await response.json();
      setIsModelTrained(data.ratingCount >= minRequiredRatings);
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const checkUserStatus = async (id) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/${id}/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setUserId(id);
      localStorage.setItem('userId', id);
      setHasEnoughRatings(data.hasEnoughRatings);
      setStep(data.hasEnoughRatings ? 'navigation' : 'rating');
      
      await checkModelStatus(id); // Verificar estado del modelo
      
      if (!data.hasEnoughRatings) {
        await fetchInitialMovies();
      }
    } catch (error) {
      console.error('Error checking user status:', error);
      setError('Error al verificar el estado del usuario');
      setStep('user-selection');
      localStorage.removeItem('userId');
    }
  };

  const fetchInitialMovies = async () => {
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/movies/popular`);
      const data = await response.json();
      setMoviesToRate(data);
    } catch (error) {
      setError('Error al cargar películas');
    }
    setLoading(false);
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (searchType === 'title') params.append('query', searchQuery);
      if (searchType === 'tag') params.append('tag', searchQuery);
      if (searchType === 'genre') params.append('genre', searchQuery);
      
      const response = await fetch(`${API_BASE_URL}/api/movies/search/combined?${params}`);
      const data = await response.json();
      setSearchResults(data);
      setActiveTab('search'); // Asegura que estamos en la pestaña de búsqueda
    } catch (error) {
      setError('Error en la búsqueda');
    }
    setLoading(false);
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      // Primero enviamos las puntuaciones actuales si hay alguna
      if (Object.keys(userRatings).length > 0) {
        await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ratings: userRatings })
        });
      }
  
      // Luego obtenemos las nuevas recomendaciones
      const response = await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (Array.isArray(data) && data.length > 0) {
        setRecommendations(data);
      } else {
        console.warn('No se recibieron recomendaciones del servidor');
        // Si no hay recomendaciones personalizadas, obtenemos películas populares como fallback
        const popularResponse = await fetch(`${API_BASE_URL}/api/movies/popular?limit=3`);
        const popularData = await popularResponse.json();
        setRecommendations(popularData);
      }
    } catch (error) {
      console.error('Error al obtener recomendaciones:', error);
      setError('Error al obtener recomendaciones');
      // También podríamos cargar películas populares como fallback aquí
    } finally {
      setLoading(false);
    }
  };

  const loadMoreMovies = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/movies/popular`);
      const data = await response.json();
      setMoviesToRate(data);
    } catch (error) {
      setError('Error al cargar más películas');
    }
    setLoading(false);
  };
  
  const handleSubmitRatings = async () => {
    if (Object.keys(userRatings).length === 0) {
      setError('Por favor, califica al menos una película antes de continuar');
      return;
    }
  
    setLoading(true);
    try {
      // Primero enviamos las puntuaciones
      await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ratings: userRatings })
      });
  
      // Luego obtenemos las recomendaciones
      const recommendationsResponse = await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`);
      const recommendationsData = await recommendationsResponse.json();
      
      // Actualizamos el estado
      setRecommendations(recommendationsData);
      setHasEnoughRatings(true);
      setStep('navigation');
      setActiveTab('recommendations'); // Cambiamos a la pestaña de recomendaciones
      
    } catch (error) {
      console.error('Error:', error);
      setError('Error al procesar las puntuaciones y obtener recomendaciones');
    } finally {
      setLoading(false);
    }
  };

  const handleRating = async (movieId, rating) => {
    setUserRatings(prev => ({
      ...prev,
      [movieId]: rating
    }));
  };
  
  const [selectedMovieId, setSelectedMovieId] = useState(null);

  const goToRatingScreen = () => {
    if (selectedMovieId) {
      setStep('rate-recommendation');
    }
  };

  

const handleRecommendationQuality = (value) => {
  setRecommendationQuality(value);
};

const submitRecommendationFeedback = async () => {
  if (!selectedMovieId || !recommendationQuality) return;

  try {
    await fetch(`${API_BASE_URL}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        movie_id: selectedMovieId,
        feedback_type: 'recommendation',
        user_id: userId,
        quality: recommendationQuality
      })
    });
    setStep('navigation');
    setSelectedMovieId(null);
    setRecommendationQuality(null);
  } catch (error) {
    setError('Error al enviar el feedback de recomendación');
  }
};

const submitRatings = async (ratings) => {
  console.log("Enviando puntuaciones:", ratings);
  try {
    await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ratings }),
    });
    setHasEnoughRatings(true);
    setStep('navigation');
  } catch (error) {
    console.error("Error enviando puntuaciones:", error);
  }
};

  

  const handleLogout = () => {
    setUserId(null);
    localStorage.removeItem('userId');
    setStep('user-selection');
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div>
      <Header userId={userId} onLogout={handleLogout} />
      {step === 'user-selection' && (
        <div className="max-w-md mx-auto p-6">
          <Card>
            <CardHeader>
              <CardTitle>Bienvenido al Recomendador de Películas</CardTitle>
              <CardDescription>
                Ingresa tu ID de usuario o crea una nueva cuenta
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Input
                  type="number"
                  placeholder="ID de Usuario"
                  value={inputUserId}
                  onChange={(e) => setInputUserId(e.target.value)}
                />
                <Button 
                  className="w-full mt-2"
                  onClick={() => checkUserStatus(inputUserId)}
                  disabled={!inputUserId}
                >
                  Continuar
                </Button>
              </div>
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2">O</span>
                </div>
              </div>
              <Button 
                variant="outline" 
                className="w-full"
                onClick={createUser}
              >
                Crear nueva cuenta
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

{step === 'rating' && (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">
          Puntúa las películas que hayas visto
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Has puntuado {Object.keys(userRatings).length} películas
        </p>
        <div className="flex gap-4 mb-6">
          <Button 
            onClick={loadMoreMovies}
            variant="outline"
          >
            Cargar más películas
          </Button>
          <Button 
            onClick={handleSubmitRatings}
            disabled={Object.keys(userRatings).length === 0}
          >
            Enviar puntuaciones y continuar
          </Button>
        </div>
        {error && (
          <div className="text-red-500 mb-4">
            {error}
          </div>
        )}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {moviesToRate.map(movie => (
          <MovieCard
            key={movie.id}
            movie={movie}
            onRate={handleRating}
            currentRating={userRatings[movie.id]}
            setSearchQuery={setSearchQuery}
            setSearchType={setSearchType}
            handleSearch={handleSearch}
            setActiveTab={setActiveTab}
          />
        ))}
      </div>
    </div>
  )}

      {step === 'navigation' && (
        <div className="max-w-4xl mx-auto p-6">
          <div className="flex gap-4 mb-4">
            <button
              className={`px-4 py-2 border ${activeTab === 'search' ? 'bg-gray-200' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              Buscar Películas
            </button>
            <button
              className={`px-4 py-2 border ${activeTab === 'recommendations' ? 'bg-gray-200' : ''}`}
              onClick={() => setActiveTab('recommendations')}
            >
              Recomendaciones
            </button>
          </div>

          {activeTab === 'search' && (
  <Card>
    <CardHeader>
      <CardTitle>Buscar Películas</CardTitle>
    </CardHeader>
    <CardContent>
      <div className="flex gap-4 mb-6">
        <Input
          placeholder="Buscar películas..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex-1"
        />
        <select
          value={searchType}
          onChange={(e) => setSearchType(e.target.value)}
          className="px-3 py-2 border rounded-md"
        >
          <option value="title">Título</option>
          <option value="tag">Tag</option>
          <option value="genre">Género</option>
        </select>
        <Button onClick={handleSearch}>Buscar</Button>
      </div>

      <div className="flex gap-4 mb-6">
        <Button 
          onClick={async () => {
            if (Object.keys(userRatings).length === 0) {
              setError('Por favor, califica al menos una película antes de continuar');
              return;
            }

            setLoading(true);
            try {
              // Enviamos las puntuaciones
              await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ratings: userRatings })
              });

              // Obtenemos las recomendaciones
              const recommendationsResponse = await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`);
              const recommendationsData = await recommendationsResponse.json();
              
              // Actualizamos el estado
              setRecommendations(recommendationsData);
              setUserRatings({}); // Limpiamos los ratings temporales
              setActiveTab('recommendations'); // Cambiamos a la pestaña de recomendaciones
              
            } catch (error) {
              console.error('Error:', error);
              setError('Error al procesar las puntuaciones y obtener recomendaciones');
            } finally {
              setLoading(false);
            }
          }} 
          disabled={Object.keys(userRatings).length === 0}
        >
          Enviar puntuaciones
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {searchResults.map(movie => (
          <MovieCard
            key={movie.id}
            movie={movie}
            onRate={handleRating}
            currentRating={userRatings[movie.id]}
            setSearchQuery={setSearchQuery}
            setSearchType={setSearchType}
            handleSearch={handleSearch}
            setActiveTab={setActiveTab}
          />
        ))}
      </div>
    </CardContent>
  </Card>
)}
          
          {activeTab === 'recommendations' && (
      <Card>
        <CardHeader>
          <CardTitle>Recomendaciones Personalizadas</CardTitle>
          <CardDescription>
            {isModelTrained 
              ? "Haz clic para obtener tus recomendaciones personalizadas"
              : `Necesitas puntuar al menos ${minRequiredRatings} películas para obtener recomendaciones personalizadas. 
                 Actualmente tienes ${userRatings ? Object.keys(userRatings).length : 0} puntuaciones.`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {recommendations.length > 0 ? (
              recommendations.map(movie => (
                <div
                  key={movie.id}
                  role="button"
                  tabIndex={0}
                  className="cursor-pointer transition-transform hover:scale-105"
                  onClick={() => {
                    setSelectedMovie(movie);
                    setStep('rate-recommendation');
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      setSelectedMovie(movie);
                      setStep('rate-recommendation');
                    }
                  }}
                >
                  <MovieCard movie={movie} showFeedback={false} />
                </div>
              ))
            ) : (
              <div className="text-center py-8 col-span-3">
                {isModelTrained ? (
                  <Button onClick={fetchRecommendations}>
                    Obtener Recomendaciones
                  </Button>
                ) : (
                  <div className="space-y-4">
                    <Button 
                      variant="outline"
                      onClick={() => setActiveTab('search')}
                    >
                      Ir a buscar películas para puntuar
                    </Button>
                    <p className="text-sm text-gray-500">
                      Puedes obtener recomendaciones rápidas basadas en géneros similares mientras tanto
                    </p>
                    <Button onClick={fetchRecommendations}>
                      Obtener recomendaciones rápidas
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    )}
  </div>
)}

{step === 'rate-recommendation' && selectedMovie && (
  <RecommendationRatingScreen
    movie={selectedMovie}
    currentRating={currentMovieRating}
    currentQuality={recommendationQuality}
    onRating={(value) => {
      setCurrentMovieRating(value);
      console.log(`Rated movie: ${selectedMovie.id} with ${value} stars`);
    }}
    onQualityRating={(value) => {
      setRecommendationQuality(value);
      console.log(`Rated recommendation quality: ${value}`);
    }}
    onBack={() => {
      setStep('navigation');
      setSelectedMovie(null);
      setCurrentMovieRating(null);
      setRecommendationQuality(null);
    }}
    onSubmit={async () => {
      try {
        console.log(`Submitting ratings for movie ${selectedMovie.id}`);
        await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ratings: {
              [selectedMovie.id]: currentMovieRating
            }
          })
        });

        await fetch(`${API_BASE_URL}/api/feedback/recommendation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            movie_id: selectedMovie.id,
            quality: recommendationQuality
          })
        });

        setStep('navigation');
        setSelectedMovie(null);
        setCurrentMovieRating(null);
        setRecommendationQuality(null);
        await fetchRecommendations(); // Obtener nuevas recomendaciones después de enviar el feedback
      } catch (error) {
        console.error('Error submitting rating:', error);
        setError('Error al enviar la evaluación. Por favor, inténtalo de nuevo.');
      }
    }}
  />
)}



        </div>
      )}
    
  


export default OnboardingFlow;


