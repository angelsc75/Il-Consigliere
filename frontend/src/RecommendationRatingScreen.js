import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './components/ui/card';
import { Button } from './components/ui/button';
import { Star, Film } from 'lucide-react';
import Rating from 'react-rating';
const RatingStars = ({ rating, onRate, size = "w-6 h-6" }) => {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((value) => (
        <button
          key={value}
          onClick={() => onRate(value)}
          className="focus:outline-none transition-transform hover:scale-110"
          aria-label={`Rate ${value} stars`}
        >
          <Star
            className={`${size} ${
              value <= rating 
                ? 'fill-yellow-500 text-yellow-500' 
                : 'fill-none text-gray-300 hover:text-yellow-400'
            } transition-colors`}
          />
        </button>
      ))}
    </div>
  );
};

const QualityRating = ({ currentQuality, onQualityRating }) => {
  const getButtonStyle = (value) => {
    if (currentQuality === value) {
      return "bg-blue-500 text-white hover:bg-blue-600";
    }
    return "bg-white text-gray-700 hover:bg-gray-100";
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map((value) => (
          <Button
            key={value}
            onClick={() => onQualityRating(value)}
            className={`w-12 h-12 font-semibold ${getButtonStyle(value)}`}
          >
            {value}
          </Button>
        ))}
      </div>
      <div className="flex justify-between text-sm text-gray-600">
        <span>Muy mala recomendación</span>
        <span>Excelente recomendación</span>
      </div>
    </div>
  );
};

const RecommendationRatingScreen = ({ 
  movie, 
  onSubmit, 
  onBack,
  onRating,
  onQualityRating,
  currentRating,
  currentQuality
}) => {
    if (!movie) {
        console.error("La película es nula o no fue pasada correctamente:", movie);
        return <p>No se pudo cargar la película seleccionada.</p>;
    }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <Card className="w-full bg-white shadow-lg">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold">Evalúa esta recomendación</CardTitle>
          <p className="text-gray-500">Tu opinión nos ayuda a mejorar nuestras recomendaciones</p>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-gray-900">{movie.title}</h3>
            
            <div className="relative aspect-[2/3] w-full max-w-sm mx-auto">
              {movie.poster_path ? (
                <img 
                  src={movie.poster_path} 
                  alt={movie.title}
                  className="rounded-lg object-cover w-full h-full shadow-md"
                />
              ) : (
                <div className="w-full h-full bg-gray-100 rounded-lg flex items-center justify-center">
                  <Film className="w-16 h-16 text-gray-400" />
                </div>
              )}
            </div>
            
            {movie.overview && (
              <p className="text-gray-600 leading-relaxed">{movie.overview}</p>
            )}
          </div>

          <div className="space-y-8 bg-gray-50 p-6 rounded-lg">
            <div className="space-y-3">
              <h4 className="text-lg font-medium text-gray-900">¿Qué te pareció esta película?</h4>
              <RatingStars rating={currentRating} onRate={onRating} size="w-8 h-8" />
              <p className="text-sm text-gray-500">
                {currentRating ? `Tu puntuación: ${currentRating} de 5 estrellas` : 'Aún no has puntuado esta película'}
              </p>
            </div>

            <div className="space-y-3 pt-4 border-t border-gray-200">
              <h4 className="text-lg font-medium text-gray-900">¿Qué tan acertada fue esta recomendación?</h4>
              <QualityRating 
                currentQuality={currentQuality} 
                onQualityRating={onQualityRating} 
              />
            </div>
          </div>
        </CardContent>
        
        <CardFooter className="flex justify-between pt-6">
          <Button 
            variant="outline" 
            onClick={onBack}
            className="px-6"
          >
            Volver
          </Button>
          <Button 
            onClick={onSubmit}
            disabled={!currentRating || !currentQuality}
            className="px-6 bg-blue-500 hover:bg-blue-600"
          >
            Enviar evaluación
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default RecommendationRatingScreen;