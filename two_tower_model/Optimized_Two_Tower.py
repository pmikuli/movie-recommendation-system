import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from datetime import datetime
from tqdm import tqdm

import random
from pathlib import Path

from sklearn.model_selection import train_test_split
from itertools import chain

from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm, trange

import math
import faiss
import os


# ========== CLASSES ==========

class MovieSegmentation:
    def __init__(self, df_movies):
        self.df_movies = df_movies
        self.segments = self._create_segments()
        self._print_segment_stats()

    def _create_segments(self):
        df = self.df_movies.copy()

        segments = {
            'blockbuster':      df[df['if_blockbuster'] == 1].index.tolist(),                                           # 2,280
            'highly_watched':   df[df['highly_watched'] == 1].index.tolist(),                                           # 7,568
            'highly_rated':     df[df['highly_rated'] == 1].index.tolist(),                                             # 8,261
            'mainstream':       df[df['engagement_score'] >= 0.75].index.tolist(),                                      # 19,383
            'niche':            df[(df['engagement_score'] > 0) & (df['engagement_score'] < 0.75)].index.tolist(),      # 22,589
            'obscure':          df[df['engagement_score'] <= 0].index.tolist()                                          # 42,161
        }
        return segments

    def _print_segment_stats(self):
        print("--- STATYSTYKI FILMOW (Overlaps) ---")
        total_movies = len(self.df_movies)
        for segment, movies in self.segments.items():
            percentage = (len(movies) / total_movies) * 100
            print(f"{segment.upper():>15}: {len(movies):>6,} ({percentage:>5.1f}%)")

class NegativeSampler:

    def __init__(self, df_ratings, df_movies, initial_faiss_index, initial_movie_matrix, movie_to_local, local_to_movie):
        self.df_ratings = df_ratings
        self.all_movie_ids = df_movies.index.to_numpy()

        self.faiss_index = initial_faiss_index
        self.movie_matrix_np = initial_movie_matrix
        self.movie_to_local = movie_to_local
        self.local_to_movie = local_to_movie

        movie_segmentation = MovieSegmentation(df_movies)
        self.segment_pools = {
            key: np.array(val, dtype=np.int32)
            for key, val in movie_segmentation.segments.items()
        }
        print("Segment pools created.")

        self.regular_user_recipe_pct = {
            'mainstream': 0.40,
            'highly_rated': 0.20,
            'niche': 0.20,
            'blockbuster': 0.10,
            'obscure': 0.10
        }
        assert math.isclose(sum(self.regular_user_recipe_pct.values()), 1.0), "Recipe percentages must sum to 1.0"

        interaction_counts = self.df_ratings['seen'].str.len()
        heavy_user_threshold = interaction_counts.quantile(0.90)
        self.heavy_users = set(interaction_counts[interaction_counts >= heavy_user_threshold].index)
        print(f"Identified {len(self.heavy_users):,} heavy users (>= {int(heavy_user_threshold)} interactions).")

        print("Setup completed.")

    def update_faiss_index(self, new_faiss_index, new_movie_matrix):
        """
        A dedicated method to hot-swap the FAISS index and matrix during training.
        """
        print("NegativeSampler: Updating FAISS index state.")
        self.faiss_index = new_faiss_index
        self.movie_matrix_np = new_movie_matrix

    def _sample_prep_negatives(self, user_seen_array, k):
        """
        W oparciu o vektory dla szybkiego samplowania na bazie stratyfikacji
        """
        negatives = set()

        for segment, percentage in self.regular_user_recipe_pct.items():
            pool = self.segment_pools[segment]
            if len(pool) == 0:
                continue

            num_samples = int(round(k * percentage))
            if num_samples == 0:
                continue

            candidate_size = min(num_samples * 5, len(pool))
            candidates = np.random.choice(pool, size=candidate_size, replace=False)

            # Uzywamy np.isin powinno dac szybkie filtrowanie seen items
            mask = np.isin(candidates, user_seen_array, invert=True)
            valid_negs = candidates[mask]
            negatives.update(valid_negs[:num_samples])

        current_k = len(negatives)
        if current_k < k:
            needed = k - current_k
            seen_and_chosen = np.concatenate((user_seen_array, list(negatives)))
            fill_pool = np.setdiff1d(self.all_movie_ids, seen_and_chosen, assume_unique=True)

            if len(fill_pool) > 0:
                negatives.update(np.random.choice(fill_pool, size=min(needed, len(fill_pool)), replace=False))

        return list(negatives)[:k]

    def _sample_hard_negatives(self, pos_id, user_seen_array, k, top_k=200):
        """
        Sampling dla heavy users poprzez FAISS
        - k_h = int(k * hard_frac) zwraca jaka liczbe hard_neg dostarczamy
        """
        k_h = int(k * 0.5)          # Liczba hard negatywów

        hard_negs = np.array([], dtype=np.int32)

        if k_h > 0 and pos_id in self.movie_to_local:
            try:
                local_pos = self.movie_to_local[pos_id]

                _, I = self.faiss_index.search(self.movie_matrix_np[local_pos].reshape(1, -1), top_k)

                hard_cand_mask = np.isin(I[0], user_seen_array, invert=True)
                hard_cands = I[0][hard_cand_mask]

                if len(hard_cands) > 0:
                    num_to_sample = min(k_h, len(hard_cands))
                    hard_negs = np.random.choice(hard_cands, size=num_to_sample, replace=False)

            except Exception as e:
                 print(f"Blad hard negative sampling dla filmu {pos_id}: {e}")

        needed_random = k - len(hard_negs)
        if needed_random > 0:
            seen_and_hard = np.concatenate((user_seen_array, hard_negs))
            random_pool = np.setdiff1d(self.all_movie_ids, seen_and_hard, assume_unique=True)

            if len(random_pool) > 0:
                num_to_sample = min(needed_random, len(random_pool))
                random_negs = np.random.choice(random_pool, size=num_to_sample, replace=False)
                return np.concatenate((hard_negs, random_negs)).tolist()

        return hard_negs.tolist()

    def sample(self, user_id, pos_id, k):
        """
        Poprawnie wybiera metode samplowania wzgledem usera (tylko do tej sie odwolujemy)
        """
        user_seen_array = self.df_ratings.loc[user_id, 'seen']

        if user_id in self.heavy_users:
            return self._sample_hard_negatives(pos_id, user_seen_array, k)
        else:
            return self._sample_prep_negatives(user_seen_array, k)

class MovieDataset(Dataset):
    '''
    Potrzebny do stworzenia matrix-a pod LOOCV
    '''
    def __init__(self, df_movies, max_len_a, max_len_d, max_len_g):
        self.df = df_movies
        self.max_len_a = max_len_a
        self.max_len_d = max_len_d
        self.max_len_g = max_len_g
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        m = self.df.iloc[idx]
        return collect_movie_features(m, self.max_len_a, self.max_len_d, self.max_len_g)

class TwoTowerDataset(Dataset):

    def __init__(self, df_users, df_ratings, df_movies, negative_sampler, max_len_a, max_len_d, max_len_g, k_negatives=50):
        self.df_users = df_users.reset_index(drop=True)
        self.df_ratings = df_ratings
        self.df_movies = df_movies
        self.k_negatives = k_negatives

        self.max_len_a = max_len_a
        self.max_len_d = max_len_d
        self.max_len_g = max_len_g

        self.negative_sampler = negative_sampler

    def __len__(self):
        return len(self.df_users)

    def __getitem__(self, idx):
        # --- USER FEATURES ---
        u_row = self.df_users.iloc[idx]
        movies_seq, ratings_seq, ts_seq, user_stats = collect_user_features(u_row)
        user_id = u_row['userId']

        user_data = self.df_ratings.loc[user_id]
        pos_list = user_data['pos']
        # seen_set = set(user_data['seen'])

        if not pos_list:
            raise ValueError(f"Użytkownik {user_id} nie ma pozytywnych ratingów!")

        # --- BPR ---
        pos_id = random.choice(pos_list)
        neg_ids = self.negative_sampler.sample(user_id, pos_id, self.k_negatives)

        # --- DEBUG ---
        assert pos_id not in neg_ids,                       f"Wylosowałeś negatyw równy pozytywowi {user_id}!"
        assert len(neg_ids) == self.k_negatives,            f"Zła liczba negatywów {len(neg_ids)} != {self.k_negatives}"
        # assert all(nid not in seen_set for nid in neg_ids), f"Negatyw był już widziany przez użytkownika {user_id}!"

        # --- COLLECT ITEMS ---
        m_pos = self.df_movies.loc[pos_id]
        pos_feats, pos_text, pos_actors, pos_directors, pos_genres = collect_movie_features(m_pos, self.max_len_a, self.max_len_d, self.max_len_g)

        neg_feats_list, neg_text_list, neg_actor_list, neg_director_list, neg_genre_list = [], [], [], [], []
        for nid in neg_ids:
            m_neg = self.df_movies.loc[nid]
            nf, nt, na, nd, ng = collect_movie_features(m_neg, self.max_len_a, self.max_len_d, self.max_len_g)
            neg_feats_list.append(nf)
            neg_text_list.append(nt)
            neg_actor_list.append(na)
            neg_director_list.append(nd)
            neg_genre_list.append(ng)

        return {
            'user': {
                'user_statistics': user_stats,
                'movies': movies_seq,
                'ratings': ratings_seq,
                'times': ts_seq,
            },
            'pos_item': {
                'dense_features': pos_feats,
                'text_embedding': pos_text,
                'actor_ids': pos_actors,
                'director_ids': pos_directors,
                'genre_ids': pos_genres,
            },
            'neg_item': {
                'dense_features':  torch.stack(neg_feats_list),    # [k, dense_feat_dim]
                'text_embedding':  torch.stack(neg_text_list),     # [k, text_emb_dim]
                'actor_ids':       torch.stack(neg_actor_list),    # [k, max_len_a]
                'director_ids':    torch.stack(neg_director_list), # [k, max_len_d]
                'genre_ids':       torch.stack(neg_genre_list),    # [k, max_len_g]
            }
        }


class ValidationDataset(Dataset):
    """
    Pod szybka walidacje ROC/AUC
    """
    def __init__(self, df_users, df_ratings, df_movies, sampler, max_len_a, max_len_d, max_len_g, k_negatives=100):
        self.df_users = df_users.reset_index(drop=True)
        self.df_movies = df_movies
        self.k_negatives = k_negatives

        self.max_len_a = max_len_a
        self.max_len_d = max_len_d
        self.max_len_g = max_len_g

        print("ValidationDataset: Pre-sampling all negatives...")
        self.fixed_negatives = {}

        for user_id in tqdm(self.df_users['userId'], desc="Pre-sampling negatives"):
            user_data = df_ratings.loc[user_id]
            pos_list = user_data['pos']
            if not pos_list: continue

            pos_id = pos_list[0]                # Uzywamy dla powtarzalnych wynikow
            self.fixed_negatives[user_id] = {
                'pos': pos_id,
                'neg': sampler.sample(user_id, pos_id, k_negatives)
            }
        print("Pre-sampling complete.")

    def __len__(self):
        return len(self.df_users)

    def __getitem__(self, idx):
        u_row = self.df_users.iloc[idx]
        movies_seq, ratings_seq, ts_seq, user_stats = collect_user_features(u_row)
        user_id = u_row['userId']

        pos_id = self.fixed_negatives[user_id]['pos']
        neg_ids = self.fixed_negatives[user_id]['neg']

        # --- COLLECT ITEMS ---
        m_pos = self.df_movies.loc[pos_id]
        pos_feats, pos_text, pos_actors, pos_directors, pos_genres = collect_movie_features(m_pos, self.max_len_a,self.max_len_d,self.max_len_g)

        neg_feats_list, neg_text_list, neg_actor_list, neg_director_list, neg_genre_list = [], [], [], [], []
        for nid in neg_ids:
            m_neg = self.df_movies.loc[nid]
            nf, nt, na, nd, ng = collect_movie_features(m_neg, self.max_len_a, self.max_len_d, self.max_len_g)
            neg_feats_list.append(nf)
            neg_text_list.append(nt)
            neg_actor_list.append(na)
            neg_director_list.append(nd)
            neg_genre_list.append(ng)

        return {
            'user': {
                'user_statistics': user_stats,
                'movies': movies_seq,
                'ratings': ratings_seq,
                'times': ts_seq,
            },
            'pos_item': {
                'dense_features': pos_feats,
                'text_embedding': pos_text,
                'actor_ids': pos_actors,
                'director_ids': pos_directors,
                'genre_ids': pos_genres,
            },
            'neg_item': {
                'dense_features': torch.stack(neg_feats_list),  # [k, dense_feat_dim]
                'text_embedding': torch.stack(neg_text_list),   # [k, text_emb_dim]
                'actor_ids': torch.stack(neg_actor_list),       # [k, max_len_a]
                'director_ids': torch.stack(neg_director_list), # [k, max_len_d]
                'genre_ids': torch.stack(neg_genre_list),       # [k, max_len_g]
            }
        }

class UserTower(nn.Module):
    def __init__(self, input_dim, n_items, embedding_dim=64):
        '''
        input_dim - the number of columns in user features, without sequence columns
        '''
        super().__init__()

        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # A layer to project rating and timestamp into a scalar weight
        self.rating_proj = nn.Linear(2, 1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, batch):
        # Embed movieIds liked by user
        m = self.item_emb(batch['movies'])

        # Get weights
        x = torch.stack([batch['ratings'], batch['times']], dim=-1)  # [B, L_u, 2]
        w = torch.sigmoid(self.rating_proj(x))

        # weighted mean-pool
        pooled = (m * w).sum(1) / w.sum(1).clamp_min(1e-6)  # [B, D]

        input = torch.cat([batch['user_statistics'], pooled], dim=-1)  # [B, stats+EMB_DIM]
        output = self.mlp(input)  # [B, EMB_DIM]
        u = F.normalize(output, dim=1)
        return u

class ItemTower(nn.Module):
    def __init__(self, dense_feat_dim, text_emb_dim, vocab_sizes, embedding_dim=64):
        '''
        vocab_sizes - tuple odpowiednio n_actors, n_directors, n_genres
        dense_feat_dim – wymiary numeric+binary+decades+text
        tex_emb_dim - Wektor o wielkosc 300 opisujacy dane tekstowe filmu
        '''
        super().__init__()

        self.actor_emb = nn.Embedding(vocab_sizes[0], embedding_dim)
        self.director_emb = nn.Embedding(vocab_sizes[1], embedding_dim)
        self.genre_emb = nn.Embedding(vocab_sizes[2], embedding_dim)

        self.meta_mlp = nn.Sequential(
            nn.Linear(dense_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

        self.text_mlp = nn.Sequential(  # --- to consider za ostre zejscie z 512 -> 64, moze posredni 256
            nn.Linear(text_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

        MLP_INPUT_DIM = embedding_dim * 5  # odpowiednio nn.Embeedings * 3 oraz meta_mlp oraz text_mlp
        self.final_mlp = nn.Sequential(
            nn.Linear(MLP_INPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, batch, key: str = "pos_item"):
        dense_feats = batch[key]['dense_features']  # [B, dense_feat_dim]
        text_emb = batch[key]['text_embedding']  # [B, text_emb_dim]

        actor_ids = batch[key]['actor_ids']  # [B, max_len_a]
        director_ids = batch[key]['director_ids']
        genre_ids = batch[key]['genre_ids']

        if dense_feats.dim() == 3:
            B, k, Z = dense_feats.shape

            # flattenujemy
            dense_flat = dense_feats.view(B * k, Z)
            text_flat = text_emb.view(B * k, -1)
            actor_flat = actor_ids.view(B * k, -1)
            director_flat = director_ids.view(B * k, -1)
            genre_flat = genre_ids.view(B * k, -1)

            # złożony batch
            flat_batch = {
                key: {
                    'dense_features': dense_flat,
                    'text_embedding': text_flat,
                    'actor_ids': actor_flat,
                    'director_ids': director_flat,
                    'genre_ids': genre_flat,
                }
            }

            emb_flat = self.forward(flat_batch, key)  # rekurencyjnie batch na embeddingi [B*k, D]

            return emb_flat.view(B, k, -1)  # [B, k, D]

        dense_vec = self.meta_mlp(dense_feats)  # [B, D]
        text_vec = self.text_mlp(text_emb)  # [B, D]

        cast_imp = dense_feats[:, 2:3]  # [B, 1]
        director_score = dense_feats[:, 3:4]  # [B, 1]

        a = self.actor_emb(actor_ids).mean(dim=1)  # [B, D]
        d = self.director_emb(director_ids).mean(dim=1)  # [B, D]
        g = self.genre_emb(genre_ids).mean(dim=1)  # [B, D]

        # We add weights based on importance score
        a = a * cast_imp
        d = d * director_score  # --- do rozwazenia Max pooling lub Attention pooling

        input = torch.cat([a, d, g, dense_vec, text_vec], dim=-1)  # [B, 5D]
        output = self.final_mlp(input)  # [B, D]
        i = F.normalize(output, dim=1)
        return i

class TwoTowerModel(nn.Module):
    def __init__(self, stats_dim, n_items, vocab_sizes,
                 dense_feat_dim, text_emb_dim, embedding_dim=64):
        super().__init__()
        self.user_tower = UserTower(stats_dim, n_items, embedding_dim)
        self.item_tower = ItemTower(dense_feat_dim, text_emb_dim, vocab_sizes, embedding_dim)

    def forward(self, batch):
        u = self.user_tower(batch['user'])
        i_pos = self.item_tower(batch, key="pos_item")
        i_neg = self.item_tower(batch, key="neg_item")

        if i_neg.dim() == 2:
            return u, i_pos, i_neg  # każdy [B, 64]

        B, k, D = i_neg.shape

        i_neg_flat = i_neg.reshape(B * k, D)  # Splaszczamy

        u_flat = u.unsqueeze(1).expand(B, k, D).reshape(B * k, D)
        pos_flat = i_pos.unsqueeze(1).expand(B, k, D).reshape(B * k, D)

        return u_flat, pos_flat, i_neg_flat

# ========== FUNCTIONS ==========

def collect_user_features(u):
    """
    Zwraca cztery tensory: movies_seq, ratings_seq, ts_seq, user_stats
    """
    movies_seq = torch.tensor(u['movies_seq'], dtype=torch.long)
    ratings_seq = torch.tensor(u['ratings_seq'], dtype=torch.float32)
    ts_seq = torch.tensor(u['ts_seq'], dtype=torch.float32)

    stats_cols = [c for c in u.index if
                  c.startswith(('num_rating', 'avg_rating', 'weekend_watcher', 'genre_', 'type_of_viewer_'))]
    user_stats = torch.tensor(u[stats_cols]
                              .astype('float32').values, dtype=torch.float32)

    return movies_seq, ratings_seq, ts_seq, user_stats

def collect_movie_features(m, max_len_a, max_len_d, max_len_g):
    """
    Zwraca cztery tensory: combined, actor_ids, director_ids, genre_ids
    """
    numeric = [
        m.runtime,
        m.engagement_score,
        m.cast_importance,
        m.director_score,
    ]
    binary = [
        m.if_blockbuster,
        m.highly_watched,
        m.highly_rated,
        m.has_keywords,
        m.has_cast,
        m.has_director,
    ]
    decades = (m[[c for c in m.index if c.startswith('decade_')]]
               .astype(int)
               .tolist())

    dense_feats = torch.tensor(numeric + binary + decades, dtype=torch.float32)
    text_emb = torch.tensor(m.text_embedded, dtype=torch.float32)

    def pad(seq, L):
        seq_list = list(seq) if not isinstance(seq, list) else seq
        padded = seq_list[:L] + [0] * max(0, L - len(seq_list))
        return torch.tensor(padded, dtype=torch.long)

    actor_ids = pad(m.actor_ids, max_len_a)
    director_ids = pad(m.director_ids, max_len_d)
    genre_ids = pad(m.genre_ids, max_len_g)

    return dense_feats, text_emb, actor_ids, director_ids, genre_ids

def build_faiss_index_for_movies(df_movies, max_len_a, max_len_d, max_len_g):
    '''
    Do poczatkowego zbudowania macierzy embeedingow dla FAISS, do szukania najblizszych sasiadow
    '''
    movie_vecs = []
    movie_ids = []

    for i, m_id in enumerate(df_movies.index):
        try:
            dense_feats, text_emb, *_ = collect_movie_features(
                df_movies.loc[m_id],
                max_len_a, max_len_d, max_len_g
            )
            combined = torch.cat([dense_feats, text_emb], dim=0)
            # normalizujemy L2 na potrzeby FAISS cosinusowego (wyplaszczanie)
            normalized_vec = F.normalize(combined, dim=0)
            movie_vecs.append(normalized_vec)
            movie_ids.append(m_id)

            if (i + 1) % 10000 == 0:
                    print(f" - Przetworzono {i + 1}/{len(df_movies)} filmów")

        except Exception as e:
            print(f" Blad przy przetwarzaniu filmu {m_id}: {e}")
            continue

    movie_matrix = torch.stack(movie_vecs)          # macierz [n_movies, D]
    movie_matrix_np = movie_matrix.cpu().numpy().astype('float32')

    print(f"Macierz filmow: {movie_matrix_np.shape}")

    # FAISS IP po L2-normalizacji = cosine similarity
    faiss_index = faiss.IndexFlatIP(movie_matrix_np.shape[1])
    faiss_index.add(movie_matrix_np)

    local_to_movie = {i: movie_id for i, movie_id in enumerate(movie_ids)}
    movie_to_local = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    print(f" - Liczba filmów: {faiss_index.ntotal:,}")
    print(f" - Wymiar wektora: {movie_matrix_np.shape[1]}")
    print(f" - Typ index: IndexFlatIP (cosine similarity)")

    return faiss_index, movie_matrix_np, local_to_movie, movie_to_local


def collate_movies(batch):

    dense_features_list = []
    text_embedding_list = []
    actor_ids_list = []
    director_ids_list = []
    genre_ids_list = []

    for row in batch:
        dense_features, text_embedding, actor_ids, director_ids, genre_ids = row

        dense_features_list.append(dense_features)
        text_embedding_list.append(text_embedding)
        actor_ids_list.append(actor_ids)
        director_ids_list.append(director_ids)
        genre_ids_list.append(genre_ids)

    batch_movie = {
        'pos_item': {
            'dense_features': torch.stack(dense_features_list),     # [B, dense_feat_dim]
            'text_embedding': torch.stack(text_embedding_list),     # [B, text_emb_dim]
            'actor_ids': torch.stack(actor_ids_list),               # [B, max_len_a]
            'director_ids': torch.stack(director_ids_list),         # [B, max_len_d]
            'genre_ids': torch.stack(genre_ids_list),               # [B, max_len_g]
        }
    }

    return batch_movie


def collate_TT(batch):
    '''
    Pelny batchowanie danych do uczenia
    '''
    user_movies, user_ratings, user_times, user_stats = [], [], [], []
    pos_dense, pos_text, pos_actor, pos_director, pos_genre = [], [], [], [], []
    neg_dense, neg_text, neg_actor, neg_director, neg_genre = [], [], [], [], []

    for row in batch:

        user_stats.append(row['user']['user_statistics'])
        user_movies.append(row['user']['movies'])
        user_ratings.append(row['user']['ratings'])
        user_times.append(row['user']['times'])

        pos_dense.append(row['pos_item']['dense_features'])
        pos_text.append(row['pos_item']['text_embedding'])
        pos_actor.append(row['pos_item']['actor_ids'])
        pos_director.append(row['pos_item']['director_ids'])
        pos_genre.append(row['pos_item']['genre_ids'])

        neg_dense.append(row['neg_item']['dense_features']) # [k, D_feat]
        neg_text.append(row['neg_item']['text_embedding'])  # [k, D_text]
        neg_actor.append(row['neg_item']['actor_ids'])
        neg_director.append(row['neg_item']['director_ids'])
        neg_genre.append(row['neg_item']['genre_ids'])

    batch_user = {
        'user_statistics': torch.stack(user_stats),     # [B, d_stats]
        'movies': torch.stack(user_movies),             # [B, L_u]
        'ratings': torch.stack(user_ratings),           # [B, L_u]
        'times': torch.stack(user_times),               # [B, L_u]
    }

    batch_pos_item = {
        'dense_features': torch.stack(pos_dense),       # [B, dense_feat_dim]
        'text_embedding': torch.stack(pos_text),        # [B, text_emb_dim]
        'actor_ids': torch.stack(pos_actor),            # [B, max_len_a]
        'director_ids':torch.stack(pos_director),       # [B, max_len_d]
        'genre_ids': torch.stack(pos_genre),            # [B, max_len_g]
    }

    batch_neg_item = {
        'dense_features': torch.stack(neg_dense),
        'text_embedding': torch.stack(neg_text),
        'actor_ids': torch.stack(neg_actor),
        'director_ids': torch.stack(neg_director),
        'genre_ids': torch.stack(neg_genre),
    }

    return {
      'user': batch_user,
      'pos_item': batch_pos_item,
      'neg_item': batch_neg_item
    }

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    else:
        return data

'''
Przygotowanie matrix-u do leave-one-out w celu 'score' do rankingu
'''
def compute_item_embeddings(model, movie_loader, device):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for mb in movie_loader:
            mb = to_device(mb, device)

            embs = model.item_tower(mb, key='pos_item')  # [batch_size, D]
            all_embs.append(embs)
    return torch.cat(all_embs, dim=0).cpu().numpy()  # [n_movies, D]

'''
Definicja loss-u BPR (Bayesian Personalized Ranking)
'''
def bpr_loss(u, i_pos, i_neg):
    pos = (u*i_pos).sum(1) # [B] score pozytywnych par
    neg = (u*i_neg).sum(1)
    return -torch.log(torch.sigmoid(pos-neg) + 1e-8).mean()

'''
Trenowanie jednej epoki, dodano odpowiednie inputy tez do testow i ewentualnych zmian

Obecnie:
- model: TwoTowerModel
- loader: DataLoader
- optimizer: Adam
- loss: bpr_loss
'''
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for raw in tqdm(loader, desc=f"Epoch {epoch} Training", leave=False, unit="batch"):
        batch = to_device(raw, device)
        optimizer.zero_grad()

        user_vec, pos_vec, neg_vec = model(batch) # forward do TwoTowerModel

        loss = bpr_loss(user_vec, pos_vec, neg_vec)

        loss.backward() # Backword i updatujemy parametry

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(f"Gradient norm: {total_norm:.4f}")

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss/len(loader) # Do wyliczania sredniej straty w epoce
    return epoch_loss

def compute_validation_loss(model, val_loader, device):
    """
    Oblicza validation loss na zbiorze z LOOCV
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for raw in val_loader:
            batch = to_device(raw, device)
            if batch is None: continue

            user_vec, pos_vec, neg_vec = model(batch)

            loss = bpr_loss(user_vec, pos_vec, neg_vec)
            total_loss += loss
            num_batches += 1

    avg_loss = (total_loss / num_batches).item() # Do wyliczania sredniej straty w epoce
    return avg_loss

'''
Lekka ewaluacja majaca za zadanie pokazac czy model sie uczy, niz odpowiadac jak dobrze tworzy ranking
'''
def light_evaluate(model, loader, device):
    model.eval()
    aucs, paac = [], []

    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, device)

            user_vec, pos_vec, neg_vec = model(batch)

            pos_score = (user_vec * pos_vec).sum(dim = -1) # [B]
            neg_score = (user_vec * neg_vec).sum(dim = -1)

            # ROC AUC
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            scores = torch.cat([pos_score, neg_score])
            aucs.append(roc_auc_score(labels.cpu(), scores.cpu()))

            # Pair-wise accuarcy
            paac.append((pos_score > neg_score).float().mean().item())

    return float(np.mean(aucs)), float(np.mean(paac))

'''
Dokladniejsza ewaluacja majaca odpowiedziec jak model radzi sobie z rankingiem dla danych uzytkownikow
'''
def heavy_evaluate(model,user_loader,item_embs_np,
                        train_pos_sets,test_pos,top_N, val_user_ids, device):
    model.eval()
    user_embs = []

    user_ids_from_loocv = val_user_ids

    with torch.no_grad():
        for raw in user_loader:
            batch = to_device(raw, device)

            u = model.user_tower(batch['user'])  # Skupiamy sie tylko na zebraniu embeddingow uzytkownika

            user_embs.append(u.cpu().numpy())

    user_embs = np.vstack(user_embs)    # [U-liczba uzytkownikow, D]

    assert len(user_ids_from_loocv) == user_embs.shape[0]
    recalls, mrrs, ndcgs = [], [], []

    for idx, user_id in enumerate(user_ids_from_loocv):

        vec = user_embs[idx]                # [D] wektor emb usera
        scores = item_embs_np @ vec         # [I] wektory score, do oceny czy to dziala poprawnie ? 'iloczyny skalarne'

        mask = np.zeros_like(scores, dtype=bool)
        mask[list(train_pos_sets[user_id])] = True  # Tworzymy maske do odsiania filmow ktore user juz widzial
        scores[mask] = -1e9

        ranked = np.argsort(-scores)[:top_N]        # Ranking
        true_set = set(test_pos[user_id])           # hold-out

        # Recall@K
        recalls.append(int(any(r in true_set for r in ranked)))

        # MRR@K
        rr = 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in true_set:
                rr = 1.0/rank
                break
        mrrs.append(rr)

        # nDCG@K
        relevance_scores = [1.0 if movie_idx in true_set else 0.0 for movie_idx in ranked]
        dcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevance_scores, 1) if rel > 0)

        ideal_relevance = [1.0] * min(len(true_set), top_N)
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevance, 1))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return float(np.mean(recalls)), float(np.mean(mrrs)), float(np.mean(ndcgs))

def prepare():
    DEBUG = True
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

    # ---------- DATA LOADING ----------
    BASE_DIR = Path(os.getcwd()).parent
    DATA_DIR = BASE_DIR / "data"

    df_users = pd.read_parquet(DATA_DIR / 'user_features_clean_warm.parquet')

    df_movies = pd.read_parquet(DATA_DIR / 'Movies_clean_Vec_v4_25keywords.parquet')

    df_ratings = pd.read_parquet(DATA_DIR / 'ratings_groupped_20pos.parquet')

    df_LOOCV = pd.read_parquet(DATA_DIR / 'ratings_LOOCV.parquet')

    # ---------- SPRAWDZANIE POKRYCIA MOVIE_ID ----------
    user_ids = set(df_users['userId'])
    ratings_user_ids = set(df_ratings['userId'])

    print(f"Users w ratings: {len(user_ids & ratings_user_ids):,}/{len(user_ids):,}")

    mids_pos = set(x for lst in df_ratings['pos'] for x in lst)
    mids_seen = set(x for lst in df_ratings['seen'] for x in lst)
    all_rated_movies = mids_pos | mids_seen
    available_movies = set(df_movies['movieId'])
    missing_movies = all_rated_movies - available_movies

    print(f"Pokrycie filmów:")
    print(f"Filmy w pos ratings: {len(mids_pos):,}")
    print(f"Filmy w seen ratings: {len(mids_seen):,}")
    print(f"Brakujące filmy w df_movies: {len(missing_movies):,}")

    pos_user_counts = {
        m: df_ratings['pos'].map(lambda lst: m in lst).sum()
        for m in missing_movies
    }
    seen_user_counts = {
        m: df_ratings['seen'].map(lambda lst: m in lst).sum()
        for m in missing_movies
    }

    df_missing_stats = (
        pd.DataFrame({
            'pos_users': pos_user_counts,
            'seen_users': seen_user_counts,
        })
        .sort_values(['pos_users', 'seen_users'], ascending=False)
    )
    print(df_missing_stats)

    valid_ids = set(df_movies['movieId'])
    df_ratings['pos'] = df_ratings['pos'].apply(lambda lst: [m for m in lst if m in valid_ids])
    df_ratings['seen'] = df_ratings['seen'].apply(lambda lst: [m for m in lst if m in valid_ids])

    df_ratings = df_ratings[df_ratings['pos'].map(len).gt(0) & df_ratings['seen'].map(len).gt(0)]

    df_ratings.info()

    # ---------- SETUP ID ----------
    '''
    Sanity check ratingow (powinno byc 19, poniewaz jeden w LOOCV)
    '''
    single_pos_users = (df_ratings['pos'].apply(len) < 19).sum()

    print(f"Liczba użytkowników z mniej niz 19 pozytywnymi ratingami: {single_pos_users}")

    empty_pos_ratings = df_ratings['pos'].apply(lambda x: len(x) == 0).sum()
    empty_seen_ratings = df_ratings['seen'].apply(lambda x: len(x) == 0).sum()

    if empty_pos_ratings != 0 or empty_seen_ratings != 0:
        print(f'Empty ratings: pos: {empty_pos_ratings}, seen: {empty_seen_ratings}')
        raise Exception("Users without a single pos/neg rating exist in the ratings_groupped_ids dataset")

    unique_ids = set(
        df_users['movies_seq'].explode().tolist()
        + df_ratings['pos'].explode().tolist()
        + df_ratings['seen'].explode().tolist()
        + df_LOOCV['holdout_movieId'].tolist()
    )

    print('Unique movieIds:', len(unique_ids))
    unique_ids = sorted(unique_ids)

    movieId_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    print('min idx:', min(movieId_to_idx.values()))
    print('max idx:', max(movieId_to_idx.values()))

    n_items = len(unique_ids)

    assert min(movieId_to_idx.values()) == 0
    assert max(movieId_to_idx.values()) == n_items - 1

    df_users['movies_seq'] = df_users['movies_seq'].apply(lambda lst: [movieId_to_idx[m] for m in lst])
    df_ratings['pos'] = df_ratings['pos'].apply(lambda lst: [movieId_to_idx[m] for m in lst])
    df_ratings['seen'] = df_ratings['seen'].apply(lambda lst: [movieId_to_idx[m] for m in lst])
    df_ratings = df_ratings.set_index('userId')

    df_movies = df_movies[df_movies['movieId'].isin(movieId_to_idx)].copy()
    df_movies['movieId'] = df_movies['movieId'].map(movieId_to_idx)
    df_movies = df_movies.set_index('movieId')

    df_LOOCV['holdout_movieId'] = df_LOOCV['holdout_movieId'].map(movieId_to_idx)

    assert df_users['movies_seq'].explode().max() < n_items
    assert df_ratings['pos'].explode().max() < n_items
    assert df_ratings['seen'].explode().max() < n_items

    assert df_movies.index.max() < n_items
    assert df_movies.index.notna().all()

    assert df_LOOCV['holdout_movieId'].notna().all()

    max_movie_idx = df_users['movies_seq'].explode().max()
    print("max_movie_idx =", max_movie_idx)
    print("n_items =", n_items)

    assert max_movie_idx < n_items, "Indeks filmu przekracza rozmiar embeddingu"

    def has_invalid_entries(seq_col):
        return seq_col.explode().isin([-1, np.nan, None]).any()

    print("Zawiera niepoprawne wartości (train):", has_invalid_entries(df_users['movies_seq']))

    # ---------- TESTER ----------
    if DEBUG:
        sampled_users = df_users.sample(n=10000, random_state=213).copy()

        mask = df_ratings.index.isin(sampled_users['userId'])
        sampled_ratings = df_ratings[mask].copy()

        mask_loocv = df_LOOCV['userId'].isin(sampled_users['userId'])
        sampled_loocv = df_LOOCV[mask_loocv].copy()

        # used_movie_ids = set(sampled_users['movies_seq'].explode()) \
        #                | set(sampled_ratings['pos'].explode()) \
        #                | set(sampled_ratings['seen'].explode()) \
        #                | set(sampled_loocv['holdout_movieId'])
        # sampled_movies = df_movies[df_movies.index.isin(used_movie_ids)].copy()

        df_users = sampled_users
        df_ratings = sampled_ratings
        df_LOOCV = sampled_loocv
        # df_movies = sampled_movies

    df_movies.info()
    df_ratings.info()
    df_users.info()
    df_LOOCV.info()

    # ---------- DANE POTRZEBNE GLOBALNIE ----------
    '''
    Globalny max_len
    '''
    max_len_a = int(df_movies['actor_ids'].str.len().max())
    max_len_d = int(df_movies['director_ids'].str.len().max())
    max_len_g = int(df_movies['genre_ids'].str.len().max())

    '''
    Dla nn.Embeedings -> Item Tower
    '''
    all_actor_ids = list(chain.from_iterable(df_movies['actor_ids']))
    num_actors = max(all_actor_ids) + 1

    all_director_ids = list(chain.from_iterable(df_movies['director_ids']))
    num_directors = max(all_director_ids) + 1

    all_genre_ids = list(chain.from_iterable(df_movies['genre_ids']))
    num_genres = max(all_genre_ids) + 1

    print(num_actors, num_directors, num_genres)

    return df_users, df_ratings, df_movies, df_LOOCV, movieId_to_idx, n_items, max_len_a, max_len_d, max_len_g, num_actors, num_directors, num_genres

def train(df_users, df_ratings, df_movies, df_LOOCV, movieId_to_idx, n_items, max_len_a, max_len_d, max_len_g, num_actors, num_directors, num_genres):
    # ---------- PRZYGOTOWANIE INDEKSU FAISS ----------
    initial_faiss_index, initial_movie_matrix_np, initial_local_to_movie, initial_movie_to_local = build_faiss_index_for_movies(
        df_movies, max_len_a, max_len_d, max_len_g)
    '''
    Przepisanie poczatkowego - FAISS index
    '''
    faiss_index = initial_faiss_index
    movie_matrix_np = initial_movie_matrix_np
    local_to_movie = initial_local_to_movie
    movie_to_local = initial_movie_to_local

    # ---------- WSTEPNE ZMIENNE ----------
    '''
    Wielkosc batcha zalezna od pamieci GPU
    '''
    BATCH_SIZE = 2048
    EMB_DIM = 64
    EPOCHS = 50
    TOP_N = 20

    '''
    Early stopping
    '''
    best_rank = 0.0  # dla metryk, które chcemy maksymalizować (np. ROC-AUC)
    epochs_no_improve = 0
    patience = 4  # maksymalna liczba epok bez poprawy
    save_path = "best_model.pt"  # gdzie będziemy dumpować najlepszy model

    '''
    Przygotowanie workerow do treningu
    '''
    num_workers_prep = 4            #os.cpu_count() // 2 => 8
    print(f"Using {num_workers_prep} workers for DataLoaders.")

    sampler = NegativeSampler(
        df_ratings,
        df_movies,
        initial_faiss_index,
        initial_movie_matrix_np,
        movie_to_local,
        local_to_movie
    )

    # ---------- TEST DATASETU ----------
    '''
    TEST DATASETU I ODPOWIEDNIEGO OUTPUTU POJEDYNCZEGO OBIEKTU GET_ITEM
    '''

    dataset_test = TwoTowerDataset(df_users, df_ratings, df_movies, sampler, max_len_a, max_len_d, max_len_g)

    sample0 = dataset_test[0]

    print("Keys:", sample0.keys())
    print("\n--- USER ---")
    for k, v in sample0['user'].items():
        print(f" user[{k}]:", type(v), getattr(v, "shape", v[:5] if isinstance(v, list) else v))

    print("\n--- POS ITEM ---")
    for k, v in sample0['pos_item'].items():
        print(f" pos_item[{k}]:", type(v), v.shape if hasattr(v, 'shape') else v[:5])

    print("\n--- NEG ITEM ---")
    for k, v in sample0['neg_item'].items():
        print(f" neg_item[{k}]:", type(v), v.shape if hasattr(v, 'shape') else v[:5])

    # ---------- PRZYGOTOWANIE DATALOADEROW ----------
    '''
    Wczytanie danych do treningu
    '''
    train_dataset = TwoTowerDataset(
        df_users,
        df_ratings,
        df_movies,
        negative_sampler=sampler,
        k_negatives=50,
        max_len_a=max_len_a,
        max_len_d=max_len_d,
        max_len_g=max_len_g
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers_prep,
        pin_memory=True,
        collate_fn=collate_TT,
        drop_last=False,
        persistent_workers = True if num_workers_prep > 0 else False
    )

    '''
    Wczytanie danych ewaluacyjnych
    '''
    val_user_ids = df_LOOCV['userId'].tolist()

    val_dataset = ValidationDataset(
        df_users,
        df_ratings,
        df_movies,
        sampler=sampler,
        k_negatives=100,
        max_len_a=max_len_a,
        max_len_d=max_len_d,
        max_len_g=max_len_g
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers_prep,
        pin_memory=True,
        collate_fn=collate_TT,
        drop_last=False,
        persistent_workers = True if num_workers_prep > 0 else False
    )

    test_pos_loocv = {
        u: [movie_to_local[mid]]
        for u, mid in df_LOOCV.set_index('userId')['holdout_movieId'].items()
    }

    train_pos_sets = {
        u: {movie_to_local[mid] for mid in pos_list}
        for u, pos_list in df_ratings['pos'].items()
    }

    print(f"Przygotowano dane do LOOCV:")
    print(f"Użytkowników z holdout: {len(test_pos_loocv):,}")
    print(f"Użytkowników z pozytywami: {len(train_pos_sets):,}")

    '''
    Do wczytania i obliczania item embeedings
    '''
    movie_loader = DataLoader(
        MovieDataset(df_movies, max_len_a, max_len_d, max_len_g),
        batch_size = 8192,
        collate_fn = collate_movies
    )

    # ---------- TRENING ZASADNICZY ----------
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    print('Device:', device)

    model = (TwoTowerModel(stats_dim=25,
                           n_items=n_items,
                           vocab_sizes=(num_actors, num_directors, num_genres),
                           dense_feat_dim=24,
                           text_emb_dim=300,
                           embedding_dim=EMB_DIM)
             .to(device))
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) # zmieniamy LR zgodnie z kosinusem (powinno stabilizowac trening)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)

    for epoch in trange(1, EPOCHS + 1, desc="Epochs"):

        tr_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)  # Logika treningu i Train loss
        val_loss = compute_validation_loss(model, val_loader, device)  # Val loss

        print(f"Epoch {epoch:2d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
        if epoch % 2 == 0:
            auc, pair_acc = light_evaluate(model, val_loader, device)
            print(f"LIGHT eval | val ROC-AUC={auc:.4f} | pair-acc={pair_acc:.4f}")

        if epoch % 3 == 0:

            learned_movie_matrix_np = compute_item_embeddings(model,
                                                              movie_loader, device)  # [n_movies, D] wyliczamy embeedingi filmow
            D = learned_movie_matrix_np.shape[1]

            print(f"Learned embedding dimension (D) is: {D}")
            print(f"Shape of the new matrix to be added: {learned_movie_matrix_np.shape}")

            learned_faiss_index = faiss.IndexFlatIP(D)  # Nowy indeks pod FAISS
            learned_faiss_index.add(learned_movie_matrix_np)

            sampler.update_faiss_index(learned_faiss_index, learned_movie_matrix_np)
            print("FAISS index updated with learned embeddings.")

            recall, mrr, ndcg = heavy_evaluate(
                model,
                val_loader,                 # loader zwracający tylko user embeddings
                sampler.movie_matrix_np,    # matrix do score-a
                train_pos_sets,
                test_pos_loocv,
                top_N=TOP_N,
                val_user_ids = val_user_ids,
                device=device
            )
            print(
                f"HEAVY eval | @K={TOP_N}: Recall@{TOP_N}={recall:.4f}, MRR@{TOP_N}={mrr:.4f}| nDCG@{TOP_N}={ndcg:.4f}")

            scheduler.step(ndcg)

            if ndcg > best_rank + 1e-4:
                best_rank = ndcg
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
                print(f"  poprawa! zapisano model (nDCG@{TOP_N}={best_rank:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  brak poprawy ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping — przez {patience} epok nie było lepszego nDCG@{TOP_N}.")
            break

    model.load_state_dict(torch.load(save_path))
    model.eval()

if __name__ == '__main__':
    df_users, df_ratings, df_movies, df_LOOCV, movieId_to_idx, n_items, max_len_a, max_len_d, max_len_g, num_actors, num_directors, num_genres = prepare()

    train(df_users, df_ratings, df_movies, df_LOOCV, movieId_to_idx, n_items, max_len_a, max_len_d, max_len_g, num_actors, num_directors, num_genres)


