import requests
import os
import csv
import time

#  TMDb API Config
API_KEY = '3ce20901aa140a253cb8d7d36b7836a0'
BASE_URL = 'https://api.themoviedb.org/3'

#  Ensure folder exists
os.makedirs("data", exist_ok=True)

CSV_FILE = "data/movies.csv"
TARGET_ROWS = 50000  # Fetch 50,000 movies
MOVIES_PER_PAGE = 20  # TMDb returns 20 movies per page
TOTAL_PAGES = TARGET_ROWS // MOVIES_PER_PAGE  # Approx. 2500 pages

def fetch_movies(page):
    """Fetch a page of movies."""
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"⚠ Error fetching movies (Page {page}):", response.text)
        return []
    return response.json().get('results', [])

def fetch_movie_details(movie_id):
    """Fetch detailed info for a specific movie."""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"⚠ Error fetching details for movie {movie_id}")
        return None
    return response.json()

def save_movies_to_csv():
    """Fetch movies starting from movie 10001 and save to CSV."""
    total_movies = 0
    start_movie = 10001  # Start from the 10001st movie
    start_page = (start_movie - 1) // MOVIES_PER_PAGE + 1  # Calculate the page to start from based on the movie number

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "title", "overview", "genres", "release_date", "budget", "revenue", "language", "popularity", "runtime", "production_companies"])

        for page in range(start_page, TOTAL_PAGES + 1):  # Loop through pages starting from the calculated start page
            movies = fetch_movies(page)
            if not movies:
                break  # Stop if API fails

            for movie in movies:
                if total_movies >= TARGET_ROWS:
                    print("✅ Reached 50,000 movies!")
                    return

                movie_id = movie.get("id")
                details = fetch_movie_details(movie_id)
                if not details:
                    continue

                title = details.get("title", "N/A")
                overview = details.get("overview", "N/A")
                genres = ", ".join([genre["name"] for genre in details.get("genres", [])])
                release_date = details.get("release_date", "N/A")
                budget = details.get("budget", 0)
                revenue = details.get("revenue", 0)
                language = details.get("original_language", "N/A")
                popularity = details.get("popularity", 0)
                runtime = details.get("runtime", 0)
                production_companies = ", ".join([company["name"] for company in details.get("production_companies", [])])

                writer.writerow([movie_id, title, overview, genres, release_date, budget, revenue, language, popularity, runtime, production_companies])

                total_movies += 1

            print(f"✅ Fetched {total_movies} movies so far...")
            time.sleep(0.3)  # 🔹 Delay to prevent rate limiting

    print(f"✅ Data saved in {CSV_FILE}")

# 🔹 Run script
save_movies_to_csv()