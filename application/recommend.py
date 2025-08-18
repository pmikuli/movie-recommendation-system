import joblib
import pandas as pd

scaler = joblib.load('../data/ratings-scaler.joblib')
columns_to_normalize = ['num_rating', 'avg_rating', 'genre_Action','genre_Adventure','genre_Animation','genre_Comedy','genre_Crime','genre_Documentary','genre_Drama','genre_Family','genre_Fantasy','genre_History','genre_Horror','genre_Musical','genre_Mystery','genre_Romance','genre_Science Fiction','genre_TV Movie','genre_Thriller','genre_War','genre_Western']
genre_columns = ['genre_Action','genre_Adventure','genre_Animation','genre_Comedy','genre_Crime','genre_Documentary','genre_Drama','genre_Family','genre_Fantasy','genre_History','genre_Horror','genre_Musical','genre_Mystery','genre_Romance','genre_Science Fiction','genre_TV Movie','genre_Thriller','genre_War','genre_Western']

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

def encode_type_of_viewer(df):
    encoded = ohe.fit_transform(df[['type_of_viewer']])
    encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['type_of_viewer']))
    encoded.index = df.index
    df = df.drop(columns=['type_of_viewer'])
    df = pd.concat([df, encoded], axis=1)

    return df

def prepare_user_data_for_rec_generation(ratings, df_movies):
    print(ratings)

    num_rating, avg_rating = len(ratings), 0
    movies_seq, ratings_seq, timestamp_seq = [], [], []

    for movieId, val in ratings.items():
        rating = val['value']
        avg_rating += rating
        # data is normalized before sending to the recommender system, therefore we set this value to 1, since it will exceed max time anyway
        timestamp = 1

        movies_seq.append(movieId)
        ratings_seq.append(rating)
        timestamp_seq.append(timestamp)

    avg_rating = avg_rating / num_rating

    idk = generate_genre_columns(ratings, df_movies, genre_columns)
    print('idk', idk)

def generate_genre_columns(
    ratings: dict,
    df_movies: pd.DataFrame,
    training_genre_cols
) -> pd.DataFrame:
    """
    Build a 1-row DF with columns like 'genre_Action', ... using the user's average
    rating per genre, based on df_movies['genres'] which is a list of strings.
    Any genre not present gets 0.0. Columns are aligned to training_genre_cols.
    """
    if training_genre_cols is None:
        training_genre_cols = [c for c in columns_to_normalize if c.startswith("genre_")]

    # 1) Ratings dict -> DataFrame
    rated = pd.DataFrame(
        [
            {"movieId": int(mid), "rating": float(v["value"])}
            for mid, v in ratings.items()
            if v and v.get("value") is not None
        ]
    )
    if rated.empty:
        return pd.DataFrame([[0.0] * len(training_genre_cols)], columns=training_genre_cols)

    # 2) Join to get genres list
    j = rated.merge(df_movies[["movieId", "genres"]], on="movieId", how="left").dropna(subset=["genres"])
    if j.empty:
        return pd.DataFrame([[0.0] * len(training_genre_cols)], columns=training_genre_cols)

    # 3) Ensure list, explode, clean
    def _to_list(g):
        if isinstance(g, (list, tuple, set)):
            return list(g)
        if pd.isna(g):
            return []
        return [str(g)]
    j["genres"] = j["genres"].apply(_to_list)

    e = j.explode("genres")
    e["genres"] = e["genres"].astype(str).str.strip()
    e = e[e["genres"] != ""]
    if e.empty:
        return pd.DataFrame([[0.0] * len(training_genre_cols)], columns=training_genre_cols)

    # 4) Average rating per genre
    per = e.groupby("genres", as_index=False)["rating"].mean()
    per["col"] = "genre_" + per["genres"]

    # 5) Map into the full training set of genre columns
    row = {c: 0.0 for c in training_genre_cols}
    for _, r in per.iterrows():
        col = r["col"]
        if col in row:  # ignore genres unseen during training
            row[col] = float(r["rating"])

    return pd.DataFrame([row], columns=training_genre_cols)