import torch
from Optimized_Two_Tower import UserTower, NegativeSampler, TwoTowerDataset, collate_TT, build_faiss_index_for_movies, prepare, to_device, collect_user_features
from torch.utils.data import DataLoader
import vectordatabase
import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(os.getcwd()).parent
DATA_DIR = BASE_DIR / 'data'
EMB_DIM = 64

def get_user_tower():
    location = 'cpu'
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        location = 'cuda'
    elif torch.mps.is_available():
        device = torch.device('mps')
        location = 'mps'
    print('Device:', device)

    stats_dim = 25
    n_items = 84133
    embedding_dim = EMB_DIM

    user_tower = UserTower(stats_dim, n_items, embedding_dim)
    user_tower.load_state_dict(torch.load('user_tower.pth', map_location=location))
    user_tower.to(device)

    return user_tower, device

def add_batch_dim(batch):
    """
    Add a batch dimension to any 1-D tensors in a dict (or nested dict).
    This is useful when you only have a single example.
    """
    if torch.is_tensor(batch):
        return batch.unsqueeze(0) if batch.dim() == 1 else batch
    elif isinstance(batch, dict):
        return {k: add_batch_dim(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [add_batch_dim(v) for v in batch]
    else:
        return batch

def generate_user_emb_and_find_recommendations(df_movies, movieIdx_to_idx, user_tower, device, u_row, df_history):
    print('============')
    print('User data for recommendation generation:')
    print(u_row)

    # convert movies_seq to idx so that UserTower can use nn.Embedding properly
    # make sure to create a new u_row so that proper ids can be used for displaying already rated movies
    u_row_idx = u_row.copy()
    u_row_idx['movies_seq'] = [movieIdx_to_idx[x] for x in u_row_idx['movies_seq']]
    movies_seq, ratings_seq, ts_seq, user_stats = collect_user_features(u_row_idx)

    network_input = {
        'user_statistics': user_stats,
        'movies': movies_seq,
        'ratings': ratings_seq,
        'times': ts_seq,
    }

    network_input = add_batch_dim(network_input)
    network_input = to_device(network_input, device)


    user_embedding = user_tower(network_input)

    print('Embedding')
    print(user_embedding)

    # Filter users recommendations, by seen movies (same as heavyEvaluate masks the seen movies)
    df_history.set_index('userId', inplace=True)
    current_userId = u_row['userId']

    try:
        seen_movie_ids = df_history.loc[current_userId, 'seen']
    except KeyError:
        print(f"Warning: User {current_userId} not found in history file. No filtering applied.")
        seen_movie_ids = []

    filter_expression = f"id not in {list(seen_movie_ids)}"
    print(f"Applying Milvus filter for {len(seen_movie_ids)} seen movies...")

    ('============')

    user_vector = user_embedding.cpu().tolist()

    vectordatabase.connect()

    neighbors = vectordatabase.find_neighbors('movies', user_vector, 20, filter_expression)

    print('Neighbors from Milvus:')
    print(neighbors)
    print(len(neighbors))
    print(len(neighbors[0]))

    for movieId in seen_movie_ids:
        row = df_movies[df_movies['movieId'] == movieId].iloc[0]
        title = row['title']
        print(f'Seen movie: movieId: {movieId}, title: {title}')

    for i, movieId in enumerate(u_row['movies_seq']):
        row = df_movies[df_movies['movieId'] == movieId].iloc[0]
        rating = u_row['ratings_seq'][i]
        title = row['title']
        print(f'Recently rated: movieId: {movieId}, title: {title}, rating: {rating}')

    recommendations = []
    for n in neighbors[0]:
        movieId = n['id']
        row = df_movies[df_movies['movieId'] == movieId].iloc[0]
        print(f"Recommendation: movieId: {movieId}, title: {row['title']}, distance: {n['distance']}")
        recommendations.append({
            'movieId': movieId,
            'title': row['title'],
            'distance': n['distance'],
            'poster_path': row['poster_path']
        })

    return recommendations

def get_movies_idx(df_users, df_ratings, df_LOOCV):
    unique_ids = set(
        df_users['movies_seq'].explode().tolist()
        + df_ratings['pos'].explode().tolist()
        + df_ratings['seen'].explode().tolist()
        + df_LOOCV['holdout_movieId'].tolist()
    )

    unique_ids = sorted(unique_ids)
    movieId_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}

    return movieId_to_idx

if __name__ == '__main__':
    user_tower, device = get_user_tower()
    print('User Tower loaded')

    df_users = pd.read_parquet(DATA_DIR / 'user_features_clean_warm.parquet')
    df_movies = pd.read_csv(DATA_DIR / 'Movies_final_ML.csv')
    df_LOOCV = pd.read_parquet(DATA_DIR / 'ratings_LOOCV.parquet')
    df_ratings = pd.read_parquet(DATA_DIR / 'ratings_groupped_20pos.parquet')

    print('Datasets loaded')

    movieId_to_idx = get_movies_idx(df_users, df_ratings, df_LOOCV)

    val_user_ids = df_LOOCV['userId'].tolist()

    userId = val_user_ids[17792]
    print('userId', userId)
    u_row = df_users[df_users['userId'] == userId].iloc[0]

    recommendations = generate_user_emb_and_find_recommendations(df_movies, movieId_to_idx, user_tower, device, u_row, df_ratings)
    
