import torch
import torch.optim as optim
import torch.nn as nn
from model_2 import ItemTower, train_model
from data_loading_2 import load_data, prepare_feature_tensor
from torch.utils.data import TensorDataset
from Evaluation_2 import evaluate_model_embeddings_full, plot_embedding_distribution
from sklearn.model_selection import train_test_split
import pandas as pd

# ====== KONFIGURACJA ======
MOVIES_PARQUET_PATH = "../Data/Movies_clean_Vec_v3.parquet"
RATINGS_PATH = "../Data/ratings_clean_groupped_not_normalized.parquet"
BATCH_SIZE = 1024
EPOCHS = 40
EMBEDDING_DIM = 128
LR = 1e-3
RATING_THRESHOLD = 4.0

def create_user_item_dict(df_ratings):
    user_item_dict = {}
    for _, row in df_ratings.iterrows():
        user_id = row['userId']
        movie_ids = row['movies_seq']
        ratings = row['ratings_seq']

        filtered = [movie_id for movie_id, rating in zip(movie_ids, ratings) if rating >= RATING_THRESHOLD]

        if len(filtered) >= 2:
            user_item_dict[user_id] = filtered

    return user_item_dict


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Device:", device)

    # ====== WCZYTANIE FILMÓW I DANYCH ======
    df_movies, df_ratings = load_data(MOVIES_PARQUET_PATH, RATINGS_PATH)
    (features_tensor, movie_id_map,
     actor_idx_bag, actor_offsets,
     director_idx_bag, director_offsets,
     genre_idx_bag, genre_offsets,
     num_actors, num_directors, num_genres) = prepare_feature_tensor(df_movies)

    print("Sample feature vector:", features_tensor[0])
    print("Has NaNs:", torch.isnan(features_tensor).any().item())
    print("Has infs:", torch.isinf(features_tensor).any().item())
    print("Min/max:", features_tensor.min().item(), features_tensor.max().item())

    print("🎭 actor_idx_bag shape:", actor_idx_bag.shape)
    print("🎬 director_idx_bag shape:", director_idx_bag.shape)
    print("🏷️ genre_idx_bag shape:", genre_idx_bag.shape)
    print("📍 Num movies:", features_tensor.shape[0])

    # ====== PODZIAŁ NA TRAIN / TEST ======
    from sklearn.model_selection import train_test_split

    # 🔁 Najpierw dzielimy po userId
    all_users = df_ratings['userId'].unique()
    train_users, test_users = train_test_split(all_users, test_size=0.3, random_state=42)

    # 🔨 Tworzymy zbiory ocen na podstawie przypisania userów
    train_df = df_ratings[df_ratings['userId'].isin(train_users)]
    test_df = df_ratings[df_ratings['userId'].isin(test_users)]

    # ✅ Budujemy słowniki tylko z userami, którzy mają co najmniej 2 oceny powyżej threshold
    train_user_item_dict = create_user_item_dict(train_df)
    test_user_item_dict = create_user_item_dict(test_df)

    print(f"✅ train_user_item_dict: {len(train_user_item_dict)} users")
    print(f"✅ test_user_item_dict: {len(test_user_item_dict)} users")

    # 🔄 mapa indeksu embeddingu -> movieId
    reverse_movie_id_map = {v: k for k, v in movie_id_map.items()}

    # 🏷️ mapa movieId -> title (np. z MovieFinal.csv)
    df_movies_final = pd.read_csv('../Data/Movies_final.csv')
    movie_id_to_title = dict(zip(df_movies_final["movieId"], df_movies_final["title"]))

    # ====== INICJALIZACJA MODELU ======
    model = ItemTower(
        input_dim=features_tensor.shape[1],
        embedding_dim=EMBEDDING_DIM,
        num_actors=num_actors,
        num_directors=num_directors,
        num_genres=num_genres
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)

    # ====== TRENING Z EWALUACJĄ ======
    trained_model = train_model(
        model=model,
        features_tensor=features_tensor,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        eval_fn=evaluate_model_embeddings_full,
        eval_data={
            "train_user_item_dict": train_user_item_dict,
            "test_user_item_dict": test_user_item_dict,
            "movie_id_map": movie_id_map,
            "ks": [5, 10, 20],
            "similarity": "cosine",
            "reverse_movie_id_map": reverse_movie_id_map,
            "movie_id_to_title": movie_id_to_title
        },
        eval_every=5,
        actor_idx_bag=actor_idx_bag,
        actor_offsets=actor_offsets,
        director_idx_bag=director_idx_bag,
        director_offsets=director_offsets,
        genre_idx_bag=genre_idx_bag,
        genre_offsets=genre_offsets,
        batch_size=BATCH_SIZE,
        train_user_item_dict=train_user_item_dict
    )
