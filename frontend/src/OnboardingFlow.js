import React, { useState, useEffect } from 'react';
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

const API_BASE_URL = 'http://localhost:8000';

const REQUIRED_RATINGS = 10;

const MovieCard = ({ movie, onRate, currentRating, showFeedback = false, onFeedback }) => {
  if (!movie) return null;
  
  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle>{movie.title}</CardTitle>
        {movie.overview && (
          <CardDescription className="line-clamp-2">{movie.overview}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {movie.poster_path ? (
          <img 
            src={movie.poster_path} 
            alt={movie.title} 
            className="w-full h-48 object-cover rounded-md"
          />
        ) : (
          <div className="w-full h-48 bg-gray-200 flex items-center justify-center rounded-md">
            <Film className="w-12 h-12 text-gray-400" />
          </div>
        )}
        <div className="mt-4 flex flex-wrap gap-2">
          {movie.tags?.map(tag => (
            <span key={tag.id} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
              {tag.name}
            </span>
          ))}
        </div>

      </CardContent>
      <CardFooter className="flex justify-center gap-2">
        {onRate && (
          <div className="flex gap-2">
            {[1, 2, 3, 4, 5].map(rating => (
              <Button
                key={rating}
                variant={currentRating === rating ? "default" : "outline"}
                onClick={() => onRate(movie.id, rating)}
                className="w-10 h-10 p-0"
              >
                {rating}
              </Button>
            ))}
          </div>
        )}
        {showFeedback && (
          <div className="flex gap-2">
            <Button 
              onClick={() => onFeedback(movie.id, 'like')}
              variant="outline"
              className="flex gap-2"
            >
              <ThumbsUp className="w-4 h-4" />
            </Button>
            <Button 
              onClick={() => onFeedback(movie.id, 'dislike')}
              variant="outline"
              className="flex gap-2"
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

  const checkUserStatus = async (id) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/${id}/status`);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setUserId(id);
      localStorage.setItem('userId', id);
      setHasEnoughRatings(data.hasEnoughRatings);
      setStep(data.hasEnoughRatings ? 'navigation' : 'rating');
      
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
    } catch (error) {
      setError('Error en la búsqueda');
    }
    setLoading(false);
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`);
      const data = await response.json();
      setRecommendations(data);
    } catch (error) {
      setError('Error al obtener recomendaciones');
    }
    setLoading(false);
  };

  const handleRating = async (movieId, rating) => {
    const updatedRatings = {
      ...userRatings,
      [movieId]: rating
    };
    setUserRatings(updatedRatings);
    
    const totalRatings = Object.keys(updatedRatings).length;
    setProgress((totalRatings / REQUIRED_RATINGS) * 100);
    
    if (totalRatings >= REQUIRED_RATINGS) {
      await submitRatings(updatedRatings);
    }
  };

  const submitRatings = async (ratings) => {
    try {
      await fetch(`${API_BASE_URL}/api/movies/recommendations/${userId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ratings })
      });
      setHasEnoughRatings(true);
      setStep('navigation');
    } catch (error) {
      setError('Error al enviar puntuaciones');
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  if (step === 'user-selection') {
    return (
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
    );
  }

  if (step === 'rating') {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">
            Puntúa estas películas para obtener recomendaciones personalizadas
          </h2>
          <Progress value={progress} className="w-full" />
          <p className="text-sm text-gray-500 mt-2">
            {Object.keys(userRatings).length} de {REQUIRED_RATINGS} películas puntuadas
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {moviesToRate.slice(0, REQUIRED_RATINGS).map(movie => (
            <MovieCard
              key={movie.id}
              movie={movie}
              onRate={handleRating}
              currentRating={userRatings[movie.id]}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
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
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {searchResults.map(movie => (
                <MovieCard
                  key={movie.id}
                  movie={movie}
                  onRate={handleRating}
                  currentRating={userRatings[movie.id]}
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
          </CardHeader>
          <CardContent>
            {recommendations.length === 0 ? (
              <div className="text-center py-8">
                <Button onClick={fetchRecommendations}>
                  Obtener Recomendaciones
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recommendations.map(movie => (
                  <MovieCard
                    key={movie.id}
                    movie={movie}
                    onRate={handleRating}
                    currentRating={userRatings[movie.id]}
                    showFeedback={true}
                  />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default OnboardingFlow;

