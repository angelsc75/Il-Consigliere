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
        {movie.posterPath && (
          <img 
            src={movie.posterPath} 
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
    posterPath: PropTypes.string,
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
        {movie.posterPath && (
          <img 
            src={movie.posterPath} 
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
    posterPath: PropTypes.string,
    overview: PropTypes.string,
    tags: PropTypes.arrayOf(PropTypes.string)
  }),
  onFeedback: PropTypes.func.isRequired
};

const OnboardingFlow = () => {
  const [step, setStep] = useState('initial');
  const [moviesToRate, setMoviesToRate] = useState([]);
  const [userRatings, setUserRatings] = useState({});
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  
  const REQUIRED_RATINGS = 5;
  
  const fetchInitialMovies = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoints.popular);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Movies received:', data);
      
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error('Invalid data format received from server');
      }
      
      const validMovies = data.map(movie => ({
        id: movie.id,
        title: movie.title,
        posterPath: movie.poster_path,
        overview: movie.overview || '',
        tags: movie.tags || []
      }));
      
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
    if (step === 'initial') {
      fetchInitialMovies();
    }
  }, [step]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoints.recommendations, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: 1,
          ratings: userRatings
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRecommendations(data.map(movie => ({
        id: movie.id,
        title: movie.title,
        posterPath: movie.poster_path,
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
    setUserRatings(prev => ({
      ...prev,
      [movieId]: rating
    }));

    const totalRatings = Object.keys(userRatings).length + 1;
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
          <p>Cargando películas...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-64 flex-col">
        <p className="text-red-500 mb-4">{error}</p>
        <Button onClick={fetchInitialMovies}>Intentar de nuevo</Button>
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