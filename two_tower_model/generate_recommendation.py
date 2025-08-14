import torch
from Optimized_Two_Tower import UserTower, NegativeSampler, TwoTowerDataset, collate_TT, build_faiss_index_for_movies, prepare, to_device, collect_user_features
from torch.utils.data import DataLoader
import vectordatabase
import pandas as pd

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
    n_items = 82779
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

def generate_user_emb_and_find_recommendations():
    user_tower, device = get_user_tower()

    num_workers_prep = 4            #os.cpu_count() // 2 => 8

    df_users, df_ratings, _, df_LOOCV, movieId_to_idx, n_items, max_len_a, max_len_d, max_len_g, num_actors, num_directors, num_genres = prepare()
    df_movies = pd.read_csv('../data/Movies_final_ML.csv')

    # idx_to_movieId = {v: k for k,v in movieId_to_idx.items()}

    val_user_ids = df_LOOCV['userId'].tolist()
    

    print('============')
    userId = val_user_ids[34]
    print('userId', userId)
    u_row = df_users[df_users['userId'] == userId].iloc[0]
    print(u_row)

    movies_seq, ratings_seq, ts_seq, user_stats = collect_user_features(u_row)

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

    ('============')

    user_vector = user_embedding.cpu().tolist()

    vectordatabase.connect()

    neighbors = vectordatabase.find_neighbors('movies', user_vector, 20)

    print('Neighbors from Milvus:')
    print(neighbors)
    print(len(neighbors))
    print(len(neighbors[0]))

    for i, movieId in enumerate(u_row['movies_seq']):
        row = df_movies[df_movies['movieId'] == movieId].iloc[0]
        rating = u_row['ratings_seq'][i]
        print(f'Previously rated: movieId: {movieId}, title: {row['title']}, rating: {rating}')

    for n in neighbors[0]:
        movieId = n['id']
        row = df_movies[df_movies['movieId'] == movieId].iloc[0]
        print(f"Recommendation: movieId: {movieId}, title: {row['title']}")

if __name__ == '__main__':
    generate_user_emb_and_find_recommendations()

