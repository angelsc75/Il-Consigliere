import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

response = requests.get(
    "https://api.themoviedb.org/3/movie/550",
    params={"api_key": api_key}
)
print(response.status_code, response.json())
