import React, { useState, useEffect } from 'react'
import { Star, ThumbsUp, ThumbsDown, Loader } from 'lucide-react'
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent, 
  CardFooter 
} from './components/ui/card';
import { Button } from "./components/ui/button"
import { Progress } from "./components/ui/progress"

const OnboardingFlow = () => {
  const [step, setStep] = useState('initial'); // initial, rating, recommendations, feedback
  const [moviesToRate, setMoviesToRate] = useState([]);
  const [userRatings, setUserRatings] = useState({});
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const REQUIRED_RATINGS = 5;

  useEffect(() => {
    fetchInitialMovies();
  }, []);

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

  const handleRating = async (movieId, rating) => {
    setUserRatings(prev => ({
      ...prev,
      [movieId]: rating
    }));

    const newProgress = (Object.keys(userRatings).length + 1) / REQUIRED_RATINGS * 100;
    setProgress(newProgress);

    if (Object.keys(userRatings).length + 1 >= REQUIRED_RATINGS) {
      await fetchRecommendations();
    }
  };

  const fetchRecommendations = async () => {
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
      setStep('recommendations');
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
    setLoading(false);
  };

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

  const MovieRatingCard = ({ movie }) => (
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
            variant={userRatings[movie.id] === rating ? "default" : "outline"}
            onClick={() => handleRating(movie.id, rating)}
            className="w-10 h-10 p-0"
          >
            {rating}
          </Button>
        ))}
      </CardFooter>
    </Card>
  );

  const RecommendationCard = ({ movie }) => (
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

  return (
    <div className="max-w-4xl mx-auto p-6">
      {step === 'rating' && (
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
              <MovieRatingCard key={movie.id} movie={movie} />
            ))}
          </div>
        </>
      )}

      {step === 'recommendations' && (
        <>
          <h2 className="text-2xl font-bold mb-8">Películas recomendadas para ti</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map(movie => (
              <RecommendationCard key={movie.id} movie={movie} />
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default OnboardingFlow;