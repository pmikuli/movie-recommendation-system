import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import torch

import os

print("--- Loading recommendation artifacts... ---")

BASE_DIR = Path(os.getcwd()).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "two_tower_model"

scaler = joblib.load(DATA_DIR / 'ratings-scaler.joblib')
age_days_scaler = joblib.load(DATA_DIR / 'age-days-scaler.joblib')
print("Scalers loaded successfully.")

df_movies_full = pd.read_csv(
    DATA_DIR / 'Movies_final_ML.csv',
    usecols=['movieId', 'genres']
)
df_movies_full['genres'] = df_movies_full['genres'].fillna('').str.split(',')
print("Full movies dataset loaded for feature generation.")

def prepare_new_user_features(ratings_dict: dict, df_movies: pd.DataFrame):
    """
    Takes raw ratings from a new user and transforms them into a single-row DataFrame
    """
    if not ratings_dict:
        raise ValueError("Ratings dictionary cannot be empty.")

    ratings_list = [
        {"movieId": mid, "rating": val['value']+1.0, "timestamp": val['timestamp']}
        for mid, val in ratings_dict.items()
    ]
    df_user = pd.DataFrame(ratings_list)

    num_rating = len(df_user)
    avg_rating = df_user['rating'].mean()
    df_user['day_of_week'] = df_user['timestamp'].dt.dayofweek
    weekend_ratio = df_user['day_of_week'].isin([5, 6]).mean()
    weekend_watcher = 1.0 if weekend_ratio > 0.5 else 0.0

    # Genre columns
    genre_cols = [col for col in scaler.feature_names_in_ if col.startswith("genre_")]

    # Join
    merged = df_user.merge(df_movies, on="movieId", how="left")
    merged = merged.explode("genres")
    merged['genres'] = merged['genres'].str.strip()

    avg_genre_ratings = merged.groupby("genres")["rating"].mean()

    # Genre, avg rating
    genre_features = {}
    for col in genre_cols:
        genre_name = col.replace("genre_", "")
        genre_features[col] = avg_genre_ratings.get(genre_name, avg_rating)

    # Type of Viewer
    if avg_rating >= 4.0:
        viewer_type = "positive"
    elif avg_rating >= 3.0:
        viewer_type = "neutral"
    else:
        viewer_type = "negative"

    type_of_viewer_positive = 1.0 if viewer_type == "positive" else 0.0
    type_of_viewer_neutral = 1.0 if viewer_type == "neutral" else 0.0
    type_of_viewer_negative = 1.0 if viewer_type == "negative" else 0.0

    # Timestamp (want last 19 most recent rated movies)
    df_user = df_user.sort_values("timestamp", ascending=False)
    df_sequence = df_user.head(19)

    movies_seq = df_sequence['movieId'].tolist()
    ratings_seq = df_sequence['rating'].tolist()

    # New age_days
    now_ts = datetime.now().timestamp()
    ts_seq_raw = ((now_ts - df_sequence['timestamp'].apply(lambda x: x.timestamp())) / 86400)
    ts_seq = age_days_scaler.transform(ts_seq_raw.values.reshape(-1, 1)).flatten().tolist()

    final_user_row = pd.DataFrame([{
        'num_rating': num_rating,
        'avg_rating': avg_rating,
        'weekend_watcher': weekend_watcher,
        **genre_features,
        'type_of_viewer_negative': type_of_viewer_negative,
        'type_of_viewer_neutral': type_of_viewer_neutral,
        'type_of_viewer_positive': type_of_viewer_positive,
        'movies_seq': movies_seq,
        'ratings_seq': ratings_seq,
        'ts_seq': ts_seq,
    }])

    # Normalizacja ostatnia
    stats_cols_to_normalize = scaler.feature_names_in_
    final_user_row[stats_cols_to_normalize] = scaler.transform(final_user_row[stats_cols_to_normalize])

    print(final_user_row.iloc[0])
    return final_user_row.iloc[0]