import torch
import pandas as pd
from pathlib import Path
from two_tower_models import ItemTower, UserTower
import os

EMB_DIM = 64

location = 'cpu'
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    locatino = 'cuda'
elif torch.mps.is_available():
    device = torch.device('mps')
    location = 'mps'
print('Device:', device)

stats_dim = 25
n_items = 82779
embedding_dim = EMB_DIM
dense_feat_dim = 24
text_emb_dim = 300
num_actors = 11465
num_directors = 5163
num_genres = 20
vocab_sizes = (num_actors, num_directors, num_genres)


# user_tower = UserTower(stats_dim, n_items, embedding_dim)
# user_tower.load_state_dict(torch.load('user_tower.pth', map_location=location))
# user_tower.to(device)
# user_tower.eval()

item_tower = ItemTower(dense_feat_dim, text_emb_dim, vocab_sizes, embedding_dim)
item_tower.load_state_dict(torch.load('item_tower.pth', map_location=location))
item_tower.to(device)
item_tower.eval()

print('Models loaded')

BASE_DIR = Path(os.getcwd()).parent
DATA_DIR = BASE_DIR / "datasets"

df_users = pd.read_parquet(DATA_DIR / 'user_features_clean_warm.parquet')

df_movies = pd.read_parquet(DATA_DIR / 'Movies_clean_Vec_v4_25keywords.parquet')

df_ratings = pd.read_parquet(DATA_DIR / 'ratings_groupped_20pos.parquet')

df_LOOCV = pd.read_parquet(DATA_DIR / 'ratings_LOOCV.parquet')

users_set = set(df_users['userId'])
loocv_set = set(df_LOOCV['userId'])

print(f"Same users? {users_set == loocv_set}")
print(f"Users not in LOOCV: {len(users_set - loocv_set)}")
print(f"LOOCV not in users: {len(loocv_set - users_set)}")

if len(users_set) == len(loocv_set) and users_set == loocv_set:
    print("df_LOOCV contains all users from df_users")
else:
    print("df_LOOCV is subset/different from df_users")

print('Datasets loaded')