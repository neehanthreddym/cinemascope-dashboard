import pandas as pd
import numpy as np
import os
import logging
import time

logger = logging.getLogger(__name__)

def load_and_process_data():
    # start_total = time.time()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_file = os.path.join(project_root, 'data', 'tmdb_movies_1990_2025.csv')
    clean_file = os.path.join(project_root, 'data', 'tmdb_movies_cleaned.csv')
    
    # --- Load raw data ---
    start = time.time()
    try:
        df = pd.read_csv(raw_file)
        logger.info(f"Loaded raw data from {raw_file}")
    except FileNotFoundError as e:
        logger.error("File not found: %s", raw_file)
        raise e
    logger.info(f"Time taken to load raw CSV: {time.time() - start:.2f} seconds")

    # --- Clean and enrich ---
    df = df.drop_duplicates()

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day

    df['id'] = df['id'].astype(str)
    
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['roi'] = (df['revenue'] - df['budget']) / df['budget']

    numeric_vars = df.select_dtypes(include=[np.number]).columns
    skewed_cols = [col for col in numeric_vars if col not in ['id', 'popularity'] and abs(df[col].skew(skipna=True)) > 1]
    for col in skewed_cols:
        norm_col = f"{col}_log"
        df[norm_col] = np.log1p(df[col].clip(lower=0))

    lang = tuple(df['original_language'].unique())
    lang_representation = (
        "English", "Japanese", "French", "Chinese(Simplified)", "German", "Spanish", "Chinese(Traditional)", "Serbo-Croatian",
        "Arabic", "Italian", "Russian", "Korean", "Persian", "Hindi", "Polish", "Telugu", "Tamil",
        "Finnish", "Greek", "Swedish", "Dutch", "Czech", "Malayalam", "Tagalog", "Kannada", "Slovak",
        "Vietnamese", "Danish", "Hungarian", "Macedonian", "Serbian", "Thai", "Norwegian", "Turkish",
        "Portuguese", "Urdu", "Hebrew", "Bengali", "Tibetan", "Bosnian", "Unknown", "Tswana", "Kurdish",
        "Romanian", "Ukrainian", "Punjabi", "Lithuanian", "Icelandic", "Indonesian", "Afrikaans",
        "Estonian", "Marathi", "Khmer", "Galician", "Sinhala", "Basque", "Azerbaijani", "Albanian",
        "Dzongkha", "Mongolian", "Irish", "Hiri Motu", "Ido", "Catalan", "Kikuyu", "Latvian", "Malay",
        "Odia", "Georgian"
    )
    language_map = dict(zip(lang, lang_representation))
    df['language_name'] = df['original_language'].map(language_map).fillna(df['original_language'])

    drop_cols = ['original_title', 'overview', 'adult', 'spoken_languages', 'original_language']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Save cleaned file
    df.to_csv(clean_file, index=False)
    logger.info(f"💾 Saved cleaned data to: {clean_file}")

    return df, skewed_cols