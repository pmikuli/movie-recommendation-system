import pandas as pd
from itertools import chain
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import faiss
import random
import torch.nn.functional as F

BASE_DIR = Path(os.getcwd()).parent
DATA_DIR = BASE_DIR / "datasets"
BATCH_SIZE = 2048

df_users = pd.read_parquet(DATA_DIR / 'user_features_clean_warm.parquet')

df_movies = pd.read_parquet(DATA_DIR / 'Movies_clean_Vec_v4_25keywords.parquet')

df_ratings = pd.read_parquet(DATA_DIR / 'ratings_groupped_20pos.parquet')

df_LOOCV = pd.read_parquet(DATA_DIR / 'ratings_LOOCV.parquet')

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

'''
Globalny max_len
'''
max_len_a = int(df_movies['actor_ids'].str.len().max())
max_len_d = int(df_movies['director_ids'].str.len().max())
max_len_g = int(df_movies['genre_ids'].str.len().max())

def collect_user_features(u):
        """
        Zwraca cztery tensory: movies_seq, ratings_seq, ts_seq, user_stats
        """
        movies_seq  = torch.tensor(u['movies_seq'], dtype=torch.long)
        ratings_seq = torch.tensor(u['ratings_seq'], dtype=torch.float32)
        ts_seq      = torch.tensor(u['ts_seq'], dtype=torch.float32)
       
        stats_cols  = [c for c in u.index if c.startswith(('num_rating','avg_rating','weekend_watcher','genre_','type_of_viewer_'))]
        user_stats  = torch.tensor(u[stats_cols]
                                        .astype('float32').values,dtype=torch.float32)

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

        actor_ids    = pad(m.actor_ids,    max_len_a)
        director_ids = pad(m.director_ids, max_len_d)
        genre_ids    = pad(m.genre_ids,    max_len_g)

        return dense_feats, text_emb, actor_ids, director_ids, genre_ids

def build_faiss_index_for_movies(df_movies):
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

initial_faiss_index, initial_movie_matrix_np, initial_local_to_movie, initial_movie_to_local = build_faiss_index_for_movies(df_movies)
faiss_index = initial_faiss_index
movie_matrix_np = initial_movie_matrix_np
local_to_movie = initial_local_to_movie
movie_to_local = initial_movie_to_local

class MovieDataset(Dataset):
    '''
    Potrzebny do stworzenia matrix-a pod LOOCV
    '''
    def __init__(self, df_movies):
        self.df = df_movies
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        m = self.df.iloc[idx]
        return collect_movie_features(m, max_len_a, max_len_d, max_len_g)

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

    def __init__(self, df_ratings, n_items):
        self.df_ratings = df_ratings
        self.n_items = n_items
        self.all_movie_ids = df_movies.index.to_numpy()

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

        if k_h > 0 and pos_id in movie_to_local:
            try:
                local_pos = movie_to_local[pos_id]

                _, I = faiss_index.search(movie_matrix_np[local_pos].reshape(1, -1), top_k)

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

class TwoTowerDataset(Dataset):

    def __init__(self, df_users, df_ratings, df_movies, k_negatives=50):
        self.df_users = df_users.reset_index(drop=True)
        self.df_ratings = df_ratings
        self.df_movies = df_movies
        self.k_negatives = k_negatives

        self.max_len_a = max_len_a
        self.max_len_d = max_len_d
        self.max_len_g = max_len_g

        self.negative_sampler = NegativeSampler(
            df_ratings=df_ratings,
            n_items=len(df_movies),
        )

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

'''
Wczytanie danych do treningu
'''
train_dataset = TwoTowerDataset(
    df_users,
    df_ratings,
    df_movies
)

train_loader = DataLoader(
    dataset       = train_dataset,
    batch_size    = BATCH_SIZE,
    shuffle       = True,
    # num_workers   = 2,
    pin_memory    = True,
    collate_fn    = collate_TT,
    drop_last     = False
)

'''
Wczytanie danych ewaluacyjnych
'''
val_user_ids = df_LOOCV['userId'].tolist()

val_dataset = TwoTowerDataset(
    df_users,
    df_ratings,
    df_movies,
    k_negatives=25
)

val_loader = DataLoader(
    dataset     = val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    pin_memory  = True,
    collate_fn  = collate_TT,
    drop_last   = False
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
    MovieDataset(df_movies),
    batch_size=8192,
    collate_fn=lambda batch: {
        'pos_item': {
            'dense_features': torch.stack([b[0] for b in batch]),
            'text_embedding': torch.stack([b[1] for b in batch]),
            'actor_ids':      torch.stack([b[2] for b in batch]),
            'director_ids':   torch.stack([b[3] for b in batch]),
            'genre_ids':      torch.stack([b[4] for b in batch]),
        }
    }
)