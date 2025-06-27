import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from dotenv import load_dotenv
import os


# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# Validate the API key
if not API_KEY:
    raise ValueError("TMDB_API_KEY not found. Please set it in the .env file.")

BASE_DISCOVER_URL = 'https://api.themoviedb.org/3/discover/movie' # Base URL for discovering movies
BASE_DETAILS_URL = 'https://api.themoviedb.org/3/movie/' # Base URL for movie details
OUTPUT_CSV = os.path.join("data", "tmdb_movies_1990_2025.csv")
LOG_FILE = os.path.join("logs", "tmdb_data_collection.log")

# --- LOGGING ---
def log_time(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + '\n')

# --- FETCH MOVIE LIST PER PAGE ---
async def fetch_movies_page(session, year, page):
    params = {
        'api_key': API_KEY,
        'primary_release_year': year,
        'sort_by': 'popularity.desc',
        'page': page
    }
    try:
        async with session.get(BASE_DISCOVER_URL, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("results", [])
            else:
                log_time(f"❌ Failed discover {year} page {page}: {resp.status}")
    except Exception as e:
        log_time(f"⚠️ Error fetching discover {year} page {page}: {e}")
    return []

# --- FETCH DETAILED INFO FOR ONE MOVIE ---
async def fetch_movie_details(session, movie_id, semaphore):
    url = f"{BASE_DETAILS_URL}{movie_id}"
    params = {'api_key': API_KEY}

    async with semaphore:
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'id': data['id'],
                        'title': data.get('title'),
                        'original_title': data.get('original_title'),
                        'overview': data.get('overview'),
                        'release_date': data.get('release_date'),
                        'original_language': data.get('original_language'),
                        'popularity': data.get('popularity'),
                        'vote_average': data.get('vote_average'),
                        'vote_count': data.get('vote_count'),
                        'adult': data.get('adult'),
                        'genres': [g['name'] for g in data.get('genres', [])],
                        'runtime': data.get('runtime'),
                        'budget': data.get('budget'),
                        'revenue': data.get('revenue'),
                        'production_countries': [c['name'] for c in data.get('production_countries', [])],
                        'spoken_languages': [l['name'] for l in data.get('spoken_languages', [])]
                    }
                else:
                    log_time(f"❌ Failed details {movie_id}: {resp.status}")
        except Exception as e:
            log_time(f"⚠️ Error fetching details for {movie_id}: {e}")
    return None

# --- MAIN SCRAPING FUNCTION ---
async def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    open(LOG_FILE, 'w').close()
    log_time("🚀 Script started")

    total_movies = []
    async with aiohttp.ClientSession() as session:
        # Step 1: Collect basic info
        log_time("🔍 Collecting basic movie data")
        for year in range(1990, 2026):
            tasks = [fetch_movies_page(session, year, page) for page in range(1, 26)]  # 25 pages per year
            results = await tqdm_asyncio.gather(*tasks, desc=f"Year {year}")
            for result in results:
                total_movies.extend(result)
            if len(total_movies) >= 20000:
                break
        log_time(f"📊 Total basic movies fetched: {len(total_movies)}")

        # Step 2: Fetch detailed info
        log_time("🔎 Enriching movie details")
        semaphore = asyncio.Semaphore(30)  # Limit concurrency to 30
        detail_tasks = [
            fetch_movie_details(session, movie['id'], semaphore)
            for movie in total_movies[:20000]
        ]
        detailed_movies = await tqdm_asyncio.gather(*detail_tasks, desc="Enriching")
        enriched = [movie for movie in detailed_movies if movie]
        log_time(f"✅ Finished enriching {len(enriched)} movies")

        # Step 3: Save to CSV
        df = pd.DataFrame(enriched)
        df.to_csv(OUTPUT_CSV, index=False)
        log_time(f"💾 Saved to: {OUTPUT_CSV}")

# --- RUN SCRIPT ---
if __name__ == "__main__":
    asyncio.run(main())