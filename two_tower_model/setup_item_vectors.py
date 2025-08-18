import torch
import pandas as pd
from pathlib import Path
from Optimized_Two_Tower import ItemTower, build_faiss_index_for_movies, NegativeSampler, MovieDataset, collate_movies, compute_item_embeddings, prepare
import os
from itertools import chain
import numpy as np
from torch.utils.data import DataLoader
import vectordatabase

EMB_DIM = 64

def get_item_tower():
    location = 'cpu'
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        location = 'cuda'
    elif torch.mps.is_available():
        device = torch.device('mps')
        location = 'mps'
    print('Device:', device)

    # stats_dim = 25
    # n_items = 82779
    embedding_dim = EMB_DIM
    dense_feat_dim = 24
    text_emb_dim = 300
    num_actors = 11606
    num_directors = 5240
    num_genres = 20
    vocab_sizes = (num_actors, num_directors, num_genres)


    # user_tower = UserTower(stats_dim, n_items, embedding_dim)
    # user_tower.load_state_dict(torch.load('user_tower.pth', map_location=location))
    # user_tower.to(device)
    # user_tower.eval()

    item_tower = ItemTower(dense_feat_dim, text_emb_dim, vocab_sizes, embedding_dim)
    item_tower.load_state_dict(torch.load('item_tower.pth', map_location=location))
    item_tower.to(device)

    return item_tower, device

def generate_embeddings_and_send_to_milvus(df_movies, idx_to_movieId):
    item_tower, device = get_item_tower()

    num_workers_prep = 4            #os.cpu_count() // 2 => 8
    print(f"Using {num_workers_prep} workers for DataLoaders.")


    movie_loader = DataLoader(
        MovieDataset(df_movies, max_len_a, max_len_d, max_len_g),
        batch_size = 8192,
        collate_fn = collate_movies,
        shuffle=False
    )

    movie_embeddings_np = compute_item_embeddings(item_tower, movie_loader, device)
    print(f"Shape of the new matrix to be added: {movie_embeddings_np.shape}")

    movie_idxs_in_order = df_movies.index.to_numpy()
    movie_ids_in_order = [idx_to_movieId[idx] for idx in movie_idxs_in_order]
    print(f"First 20 items fo movie_ids_in_order: {movie_ids_in_order[:20]}")

    assert len(movie_embeddings_np) == len(movie_ids_in_order), "Mismatch between number of embeddings and number of movie IDs!"

    vectors = [{'id': movieId, 'vector': embedding} for movieId, embedding in zip(movie_ids_in_order, movie_embeddings_np)]

    vectordatabase.connect()
    vectordatabase.insert_vector('movies', vectors)

    print('Inserted to db')

if __name__ == '__main__':
    df_users, df_ratings, df_movies, df_LOOCV, movieId_to_idx, n_items, max_len_a, max_len_d, max_len_g, num_actors, num_directors, num_genres = prepare()

    idx_to_movieId = {v: k for k,v in movieId_to_idx.items()}

    generate_embeddings_and_send_to_milvus(df_movies, idx_to_movieId)
    