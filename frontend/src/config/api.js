export const API_BASE_URL = 'http://localhost:8000';



export const endpoints = {
  popular: `${API_BASE_URL}/api/movies/popular`,
  recommendations: (userId) => `${API_BASE_URL}/api/movies/recommendations/${userId}`,
  feedback: `${API_BASE_URL}/api/feedback`,
  interactions: (userId) => `${API_BASE_URL}/api/users/${userId}/interactions`,
  search: `${API_BASE_URL}/api/movies/search`
}; 