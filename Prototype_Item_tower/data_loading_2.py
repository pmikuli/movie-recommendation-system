import pandas as pd
import numpy as np
import torch
import ast


def load_data(movies_path: str, ratings_path: str):
    df_movies = pd.read_parquet(movies_path)
    df_ratings = pd.read_parquet(ratings_path)

    if isinstance(df_ratings.iloc[0]['movies_seq'], str):
        from ast import literal_eval
        df_ratings['movies_seq'] = df_ratings['movies_seq'].apply(literal_eval)
        df_ratings['ratings_seq'] = df_ratings['ratings_seq'].apply(literal_eval)

    return df_movies, df_ratings


def prepare_feature_tensor(df_movies: pd.DataFrame):
    import ast
    df_movies = df_movies.set_index("movieId").copy()

    # === Parsuj kolumny listowe
    for col in ['text_embedded', 'genre_ids', 'actor_ids', 'director_ids']:
        df_movies[col] = df_movies[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # === Mapowania ID → indeksy
    all_actor_ids = set(i for sub in df_movies['actor_ids'] for i in sub)
    all_director_ids = set(i for sub in df_movies['director_ids'] for i in sub)
    all_genre_ids = set(i for sub in df_movies['genre_ids'] for i in sub)

    actor_id_map = {aid: idx for idx, aid in enumerate(sorted(all_actor_ids))}
    director_id_map = {did: idx for idx, did in enumerate(sorted(all_director_ids))}
    genre_id_map = {gid: idx for idx, gid in enumerate(sorted(all_genre_ids))}

    # === Tworzenie płaskich indeksów i offsetów dla EmbeddingBag
    def make_bag_inputs(id_lists, id_map):
        flat = []
        offsets = [0]
        for lst in id_lists:
            mapped = [id_map.get(i, 0) for i in lst]
            flat.extend(mapped)
            offsets.append(len(flat))
        return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets[:-1], dtype=torch.long)

    actor_idx_bag, actor_offsets = make_bag_inputs(df_movies['actor_ids'], actor_id_map)
    director_idx_bag, director_offsets = make_bag_inputs(df_movies['director_ids'], director_id_map)
    genre_idx_bag, genre_offsets = make_bag_inputs(df_movies['genre_ids'], genre_id_map)

    # === Tekstowe embeddingi
    text_tensor = np.stack(df_movies['text_embedded'].apply(np.array).to_list())

    # === Cechy numeryczne + binarne
    numeric_cols = ['runtime', 'engagement_score', 'cast_importance', 'director_score', 'release_year']
    binary_cols = ['if_blockbuster', 'highly_watched', 'highly_rated', 'has_keywords', 'has_cast', 'has_director']
    decade_cols = [col for col in df_movies.columns if col.startswith("decade_")]

    num_bin_tensor = df_movies[numeric_cols + binary_cols + decade_cols].astype(np.float32).values
    full_features = np.hstack([num_bin_tensor, text_tensor])
    features_tensor = torch.tensor(full_features, dtype=torch.float32)

    if torch.isnan(features_tensor).any():
        print("⚠️ NaN detected in feature tensor!")
        features_tensor = torch.nan_to_num(features_tensor)

    movie_id_map = {mid: idx for idx, mid in enumerate(df_movies.index)}

    return (features_tensor, movie_id_map,
            actor_idx_bag, actor_offsets,
            director_idx_bag, director_offsets,
            genre_idx_bag, genre_offsets,
            len(actor_id_map), len(director_id_map), len(genre_id_map))
