import React, { useState, useEffect } from 'react';
import { Star, ThumbsUp, ThumbsDown, Loader } from 'lucide-react';
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent, 
  CardFooter 
} from './components/ui/card';
import { Button } from "./components/ui/button";
import { Progress } from "./components/ui/progress";
import { endpoints } from './config/api';
import PropTypes from 'prop-types';

const MovieRatingCard = ({ movie, onRate, currentRating }) => {
  if (!movie) return null;
  
  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle>{movie.title}</CardTitle>
      </CardHeader>
      <CardContent>
        {movie.poster_path && (
          <img 
            src={movie.poster_path} 
            alt={movie.title} 
            className="w-full h-48 object-cover rounded-md"
          />
        )}
      </CardContent>
      <CardFooter className="flex justify-center gap-2">
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
      </CardFooter>
    </Card>
  );
};

MovieRatingCard.propTypes = {
  movie: PropTypes.shape({
    id: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    poster_path: PropTypes.string,
    overview: PropTypes.string,
    tags: PropTypes.arrayOf(PropTypes.string)
  }),
  onRate: PropTypes.func.isRequired,
  currentRating: PropTypes.number
};

const RecommendationCard = ({ movie, onFeedback }) => {
  if (!movie) return null;
  
  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle>{movie.title}</CardTitle>
      </CardHeader>
      <CardContent>
        {movie.poster_path && (
          <img 
            src={movie.poster_path} 
            alt={movie.title} 
            className="w-full h-48 object-cover rounded-md"
          />
        )}
        <p className="mt-2">{movie.overview}</p>
      </CardContent>
      <CardFooter className="flex justify-center gap-4">
        <Button 
          onClick={() => onFeedback(movie.id, 'like')}
          variant="outline"
          className="flex gap-2"
        >
          <ThumbsUp className="w-4 h-4" />
          Me gusta
        </Button>
        <Button 
          onClick={() => onFeedback(movie.id, 'dislike')}
          variant="outline"
          className="flex gap-2"
        >
          <ThumbsDown className="w-4 h-4" />
          No me gusta
        </Button>
      </CardFooter>
    </Card>
  );
};

RecommendationCard.propTypes = {
  movie: PropTypes.shape({
    id: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    poster_path: PropTypes.string,
    overview: PropTypes.string,
    tags: PropTypes.arrayOf(PropTypes.string)
  }),
  onFeedback: PropTypes.func.isRequired
};

const OnboardingFlow = () => {
  const [step, setStep] = useState('user-selection');
  const [moviesToRate, setMoviesToRate] = useState([]);
  const [userRatings, setUserRatings] = useState({});
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState(null);
  const [inputUserId, setInputUserId] = useState('');
  
  const REQUIRED_RATINGS = 5;
  

  // Load stored userId on mount
  useEffect(() => {
    const storedUserId = localStorage.getItem('userId');
    if (storedUserId) {
      setUserId(storedUserId);
      setStep('initial');
    }
  }, []);

  const handleExistingUser = async (id) => {
    if (!id) return;
    
    try {
      // Verificar si el usuario existe
      const response = await fetch(`${endpoints.users}/${id}`);
      if (response.ok) {
        setUserId(id);
        localStorage.setItem('userId', id);
        setStep('initial');
      } else {
        setError('Usuario no encontrado');
      }
    } catch (error) {
      console.error('Error verificando usuario:', error);
      setError('Error al verificar el usuario');
    }
  };

  const createUser = async () => {
    try {
      const response = await fetch(endpoints.users, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error('Error al crear usuario');
      }
      const data = await response.json();
      setUserId(data.userId);
      localStorage.setItem('userId', data.userId);
      setStep('initial');
    } catch (error) {
      console.error('Error creating user:', error);
      setError('Error al crear usuario');
    }
  };
  
  const fetchInitialMovies = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoints.popular);
      const data = await response.json();
      console.log('Movies received:', data);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      
      
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error('Invalid data format received from server');
      }
      
      
      const validMovies = data.map(movie => ({
      id: movie.id,
      title: movie.title,
      poster_path: movie.poster_path,
      overview: movie.overview,
      tags: movie.tags || []
    }));
      console.log('Processed movies:', validMovies);
      
      setMoviesToRate(validMovies);
      setStep('rating');
    } catch (error) {
      console.error('Error fetching movies:', error);
      setError(`Failed to load movies: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (step === 'initial' && userId) {
      fetchInitialMovies();
    }
  }, [step, userId]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      // Log los datos que vamos a enviar
      const requestBody = {
        ratings: userRatings
      };
      console.log('Request body:', requestBody);
  
      const response = await fetch(endpoints.recommendations(userId), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
  
      // Si hay error, vamos a ver qué dice el servidor
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Server error response:', errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      setRecommendations(data.map(movie => ({
        id: movie.id,
        title: movie.title,
        poster_path: movie.poster_path,
        overview: movie.overview || '',
        tags: movie.tags || []
      })));
      
      setStep('recommendations');
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(`Failed to load recommendations: ${error.message}`);
    }
    setLoading(false);
  };

  
  

  const handleRating = async (movieId, rating) => {
    const updatedRatings = {
      ...userRatings,
      [movieId]: rating
    };
    setUserRatings(updatedRatings);
    console.log('Updated ratings:', updatedRatings); // Para debug
  
    const totalRatings = Object.keys(updatedRatings).length;
    const newProgress = (totalRatings / REQUIRED_RATINGS) * 100;
    setProgress(newProgress);
  
    if (totalRatings >= REQUIRED_RATINGS) {
      await fetchRecommendations();
    }
  };

  const handleFeedback = async (movieId, feedbackType) => {
    try {
      await fetch(endpoints.feedback, {
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

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="flex flex-col items-center gap-4">
          <Loader className="w-8 h-8 animate-spin" />
          <p>Cargando...</p>
        </div>
      </div>
    );
  }

  if (step === 'user-selection') {
    return (
      <div className="max-w-md mx-auto p-6">
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>¿Ya tienes un ID de usuario?</CardTitle>
            <CardDescription>Introduce tu ID de usuario o crea uno nuevo</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex flex-col space-y-2">
                <label htmlFor="userId" className="text-sm font-medium">
                  ID de Usuario
                </label>
                <div className="flex gap-2">
                  <input
                    id="userId"
                    type="number"
                    value={inputUserId}
                    onChange={(e) => setInputUserId(e.target.value)}
                    className="flex-1 px-3 py-2 border rounded-md"
                    placeholder="Introduce tu ID"
                  />
                  <Button 
                    onClick={() => handleExistingUser(inputUserId)}
                    disabled={!inputUserId}
                  >
                    Continuar
                  </Button>
                </div>
              </div>
              
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-white text-gray-500">O</span>
                </div>
              </div>

              <Button 
                onClick={createUser}
                className="w-full"
                variant="outline"
              >
                Crear nuevo usuario
              </Button>
            </div>

            {error && (
              <p className="text-red-500 text-sm mt-2">{error}</p>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {step === 'rating' && moviesToRate.length > 0 && (
        <>
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-4">Puntúa estas películas</h2>
            <Progress value={progress} className="w-full" />
            <p className="text-sm text-gray-500 mt-2">
              {Object.keys(userRatings).length} de {REQUIRED_RATINGS} películas puntuadas
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {moviesToRate.slice(0, REQUIRED_RATINGS).map(movie => (
              <MovieRatingCard 
                key={movie.id} 
                movie={movie} 
                onRate={handleRating}
                currentRating={userRatings[movie.id]}
              />
            ))}
          </div>
        </>
      )}

      {step === 'recommendations' && recommendations.length > 0 && (
        <>
          <h2 className="text-2xl font-bold mb-8">Películas recomendadas para ti</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map(movie => (
              <RecommendationCard 
                key={movie.id} 
                movie={movie} 
                onFeedback={handleFeedback}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default OnboardingFlow;