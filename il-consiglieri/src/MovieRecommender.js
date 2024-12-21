import React, { useState, useEffect } from 'react';
import { Search, Star, StarHalf, Film, ThumbsUp } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";

// Constantes
const TMDB_API_KEY = '1e32e3c58214f71c5c5e035dc6f4c711';
const TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500';

// Servicio para manejar las IDs y obtener imágenes
const MovieService = {
  // Cache para los IDs
  movieLinks: new Map(),

  // Cargar el CSV de links
  async loadLinks() {
    try {
      const response = await fetch('/data/movielens/link.csv');
      const text = await response.text();
      
      text.split('\n').forEach(line => {
        const [movieId, imdbId, tmdbId] = line.split(',');
        if (movieId && tmdbId) {
          this.movieLinks.set(movieId, {
            imdbId: imdbId ? `tt${imdbId.padStart(7, '0')}` : null,
            tmdbId
          });
        }
      });
    } catch (error) {
      console.error('Error loading links:', error);
    }
  },

  // Obtener información de TMDB
  async getMovieDetails(tmdbId) {
    try {
      const response = await fetch(
        `https://api.themoviedb.org/3/movie/${tmdbId}?api_key=${TMDB_API_KEY}`
      );
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching movie details:', error);
      return null;
    }
  }
};

// Componente principal
const MovieRecommender = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  // Cargar los links al iniciar
  useEffect(() => {
    MovieService.loadLinks();
  }, []);

  // Simulación de llamada a API con imágenes
  const fetchRecommendations = async () => {
    setLoading(true);
    // Simular datos de recomendación
    const mockRecommendations = [
      { id: 1, title: 'The Shawshank Redemption', rating: 4.8, tags: ['Drama', 'Prison'], predictedRating: 4.5 },
      { id: 2, title: 'The Godfather', rating: 4.7, tags: ['Crime', 'Drama'], predictedRating: 4.3 },
      { id: 3, title: 'Inception', rating: 4.6, tags: ['Action', 'Sci-Fi'], predictedRating: 4.7 },
    ];

    // Enriquecer con datos de TMDB
    const enrichedRecommendations = await Promise.all(
      mockRecommendations.map(async (movie) => {
        const links = MovieService.movieLinks.get(movie.id.toString());
        if (links?.tmdbId) {
          const tmdbData = await MovieService.getMovieDetails(links.tmdbId);
          return {
            ...movie,
            posterPath: tmdbData?.poster_path ? 
              `${TMDB_IMAGE_BASE_URL}${tmdbData.poster_path}` :
              null,
            overview: tmdbData?.overview || ''
          };
        }
        return movie;
      })
    );

    setRecommendations(enrichedRecommendations);
    setLoading(false);
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  // Componente de Película mejorado
  const MovieCard = ({ movie }) => (
    <Card className="w-full max-w-sm hover:shadow-lg transition-shadow">
      <div className="relative aspect-[2/3] w-full">
        {movie.posterPath ? (
          <img
            src={movie.posterPath}
            alt={movie.title}
            className="absolute w-full h-full object-cover rounded-t-lg"
          />
        ) : (
          <div className="absolute w-full h-full bg-gray-200 flex items-center justify-center rounded-t-lg">
            <Film className="w-16 h-16 text-gray-400" />
          </div>
        )}
      </div>
      <CardHeader>
        <CardTitle className="text-lg font-bold">{movie.title}</CardTitle>
        <CardDescription className="line-clamp-2">
          {movie.overview}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {movie.tags.map(tag => (
              <span key={tag} className="inline-block bg-gray-200 rounded-full px-3 py-1 text-sm font-semibold text-gray-700">
                {tag}
              </span>
            ))}
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <Star className="w-5 h-5 text-yellow-400" />
              <span className="ml-1">{movie.rating}</span>
            </div>
            <div className="flex items-center">
              <StarHalf className="w-5 h-5 text-blue-500" />
              <span className="ml-1">{movie.predictedRating}</span>
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button className="w-full">
          <ThumbsUp className="w-4 h-4 mr-2" />
          Me gusta
        </Button>
      </CardFooter>
    </Card>
  );

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
          <Button>
            <Search className="w-4 h-4 mr-2" />
            Buscar
          </Button>
        </div>
      </div>

      {/* Sección de recomendaciones */}
      <div className="max-w-7xl mx-auto">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Film className="w-6 h-6 mr-2" />
          Recomendaciones para ti
        </h2>
        
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map(movie => (
              <MovieCard key={movie.id} movie={movie} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default MovieRecommender;