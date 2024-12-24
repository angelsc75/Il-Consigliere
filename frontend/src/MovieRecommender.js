import React, { useState, useEffect } from 'react'
import { Search, Star, StarHalf, Film, ThumbsUp, ThumbsDown, Loader } from 'lucide-react'
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent, 
  CardFooter 
} from './components/ui/card';
import { Input } from "./components/ui/input"
import { Button } from './components/ui/button'
import { Progress } from './components/ui/progress'

const TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500';
const REQUIRED_RATINGS = 5;

// Componente principal
const MovieRecommender = () => {
  // Estados generales
  const [hasRatedMovies, setHasRatedMovies] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  // Estados para el onboarding
  const [step, setStep] = useState('initial'); // initial, rating, recommendations
  const [moviesToRate, setMoviesToRate] = useState([]);
  const [userRatings, setUserRatings] = useState({});
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    checkUserRatings();
  }, []);

  // Verificar si el usuario ya ha calificado películas
  const checkUserRatings = async () => {
    try {
      const response = await fetch('/api/users/1/interactions');
      const interactions = await response.json();
      const hasRatings = interactions.some(i => i.rating !== null);
      setHasRatedMovies(hasRatings);
      
      if (!hasRatings) {
        await fetchInitialMovies();
      } else {
        await fetchRecommendations();
      }
    } catch (error) {
      console.error('Error checking user ratings:', error);
    }
  };

  // Obtener películas populares para el onboarding
  const fetchInitialMovies = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/movies/popular');
      const movies = await response.json();
      setMoviesToRate(movies);
      setStep('rating');
    } catch (error) {
      console.error('Error fetching initial movies:', error);
    }
    setLoading(false);
  };

  // Manejar las calificaciones durante el onboarding
  const handleInitialRating = async (movieId, rating) => {
    setUserRatings(prev => ({
      ...prev,
      [movieId]: rating
    }));

    const newProgress = (Object.keys(userRatings).length + 1) / REQUIRED_RATINGS * 100;
    setProgress(newProgress);

    if (Object.keys(userRatings).length + 1 >= REQUIRED_RATINGS) {
      await submitInitialRatings();
    }
  };

  // Enviar calificaciones iniciales y obtener recomendaciones
  const submitInitialRatings = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/movies/recommendations/1', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ratings: userRatings })
      });
      const recommendedMovies = await response.json();
      setRecommendations(recommendedMovies);
      setHasRatedMovies(true);
      setStep('recommendations');
    } catch (error) {
      console.error('Error submitting initial ratings:', error);
    }
    setLoading(false);
  };

  // Obtener recomendaciones para usuarios existentes
  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/movies/recommendations/1');
      const data = await response.json();
      setRecommendations(data);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
    setLoading(false);
  };

  // Buscar películas
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`/api/movies/search?query=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setRecommendations(data);
    } catch (error) {
      console.error('Error searching movies:', error);
    }
    setLoading(false);
  };

  // Enviar feedback sobre recomendaciones
  const handleFeedback = async (movieId, feedbackType) => {
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: 1,
          movie_id: movieId,
          feedback_type: feedbackType
        })
      });
    } catch (error) {
      console.error('Error sending feedback:', error);
    }
  };

  // Componente de tarjeta para calificación inicial
  const MovieRatingCard = ({ movie }) => (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle className="text-lg">{movie.title}</CardTitle>
      </CardHeader>
      <CardContent>
        {movie.posterPath ? (
          <img 
            src={movie.posterPath} 
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
            <span key={tag} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
              {tag}
            </span>
          ))}
        </div>
      </CardContent>
      <CardFooter className="flex justify-center gap-2">
        {[1, 2, 3, 4, 5].map(rating => (
          <Button
            key={rating}
            variant={userRatings[movie.id] === rating ? "default" : "outline"}
            onClick={() => handleInitialRating(movie.id, rating)}
            className="w-10 h-10 p-0"
          >
            {rating}
          </Button>
        ))}
      </CardFooter>
    </Card>
  );

  // Componente de tarjeta para recomendaciones
  const RecommendationCard = ({ movie }) => (
    <Card className="w-full max-w-sm hover:shadow-lg transition-shadow">
      <CardHeader>
        <CardTitle className="text-lg">{movie.title}</CardTitle>
        {movie.overview && (
          <CardDescription className="line-clamp-2">
            {movie.overview}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {movie.posterPath ? (
          <img
            src={movie.posterPath}
            alt={movie.title}
            className="w-full h-48 object-cover rounded-md mb-4"
          />
        ) : (
          <div className="w-full h-48 bg-gray-200 flex items-center justify-center rounded-md mb-4">
            <Film className="w-12 h-12 text-gray-400" />
          </div>
        )}
        <div className="flex flex-wrap gap-2 mb-4">
          {movie.tags?.map(tag => (
            <span key={tag} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
              {tag}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-4">
          {movie.rating && (
            <div className="flex items-center">
              <Star className="w-5 h-5 text-yellow-400 mr-1" />
              <span>{movie.rating.toFixed(1)}</span>
            </div>
          )}
          {movie.predictedRating && (
            <div className="flex items-center">
              <StarHalf className="w-5 h-5 text-blue-500 mr-1" />
              <span>{movie.predictedRating.toFixed(1)}</span>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-center gap-4">
        <Button 
          onClick={() => handleFeedback(movie.id, 'like')}
          variant="outline"
          className="flex gap-2"
        >
          <ThumbsUp className="w-4 h-4" />
          Me gusta
        </Button>
        <Button 
          onClick={() => handleFeedback(movie.id, 'dislike')}
          variant="outline"
          className="flex gap-2"
        >
          <ThumbsDown className="w-4 h-4" />
          No me gusta
        </Button>
      </CardFooter>
    </Card>
  );

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  // Renderizado del flujo de onboarding
  if (!hasRatedMovies) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        {step === 'rating' && (
          <>
            <div className="mb-8">
              <h2 className="text-2xl font-bold mb-4">Puntúa estas películas para obtener recomendaciones</h2>
              <Progress value={progress} className="w-full" />
              <p className="text-sm text-gray-500 mt-2">
                {Object.keys(userRatings).length} de {REQUIRED_RATINGS} películas puntuadas
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {moviesToRate.slice(0, REQUIRED_RATINGS).map(movie => (
                <MovieRatingCard key={movie.id} movie={movie} />
              ))}
            </div>
          </>
        )}
      </div>
    );
  }

  // Renderizado principal para usuarios existentes
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {/* Barra de búsqueda */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="flex gap-4">
          <Input
            type="text"
            placeholder="Buscar películas..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-grow"
          />
          <Button onClick={handleSearch}>
            <Search className="w-4 h-4 mr-2" />
            Buscar
          </Button>
        </div>
      </div>

      {/* Sección de recomendaciones */}
      <div className="max-w-7xl mx-auto">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Film className="w-6 h-6 mr-2" />
          {searchQuery ? 'Resultados de la búsqueda' : 'Recomendaciones para ti'}
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {recommendations.map(movie => (
            <RecommendationCard key={movie.id} movie={movie} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default MovieRecommender;