import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from Evaluation_2 import plot_embedding_distribution


class ItemTower(nn.Module):
    def __init__(self, input_dim, embedding_dim=64,
                 num_actors=10000, num_directors=5000, num_genres=19):
        super().__init__()
        self.actor_embedding = nn.EmbeddingBag(num_actors, 32, mode='mean')
        self.director_embedding = nn.EmbeddingBag(num_directors, 32, mode='mean')
        self.genre_embedding = nn.EmbeddingBag(num_genres, 16, mode='mean')

        self.model = nn.Sequential(
            nn.Linear(input_dim + 32 + 32 + 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x, actor_bag, actor_offsets,
                      director_bag, director_offsets,
                      genre_bag, genre_offsets):
        actor_emb = self.actor_embedding(actor_bag, actor_offsets)
        director_emb = self.director_embedding(director_bag, director_offsets)
        genre_emb = self.genre_embedding(genre_bag, genre_offsets)

        x = torch.cat([x, actor_emb, director_emb, genre_emb], dim=1)
        return self.model(x)

    def predict_embeddings(self, x, actor_bag, actor_offsets,
                                 director_bag, director_offsets,
                                 genre_bag, genre_offsets):
        with torch.no_grad():
            emb = self.forward(x, actor_bag, actor_offsets,
                                  director_bag, director_offsets,
                                  genre_bag, genre_offsets)
            norm = emb.norm(dim=1, keepdim=True)
            return emb / (norm + 1e-6)


def get_embedding_bag_inputs(indices, bag_tensor, offset_tensor):
    new_offsets = []
    new_bag = []
    offset = 0
    for i in indices:
        i = i.item()
        start = offset_tensor[i].item()
        end = offset_tensor[i + 1].item() if i + 1 < len(offset_tensor) else len(bag_tensor)
        segment = bag_tensor[start:end]
        new_bag.extend(segment.tolist())
        new_offsets.append(offset)
        offset += len(segment)
    return torch.tensor(new_bag, dtype=torch.long), torch.tensor(new_offsets, dtype=torch.long)


class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def generate_hard_triplets_faiss(embeddings, user_item_dict, movie_id_map, top_k=10,
                                  max_users=5000, max_pos_per_user=5):
    import faiss
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    triplets = []
    users = list(user_item_dict.keys())
    random.shuffle(users)

    for user in tqdm(users[:max_users], desc="🔍 Generating hard triplets with FAISS"):
        positives = user_item_dict[user]
        if len(positives) < 2:
            continue

        sampled = random.sample(positives, min(len(positives), max_pos_per_user))

        for anchor in sampled:
            for positive in sampled:
                if anchor == positive:
                    continue
                anchor_idx = movie_id_map.get(anchor)
                pos_idx = movie_id_map.get(positive)
                if anchor_idx is None or pos_idx is None:
                    continue
                D, I = index.search(embeddings_np[anchor_idx].reshape(1, -1), top_k + len(sampled))
                negatives = [i for i in I[0] if i not in [movie_id_map[m] for m in sampled] and i != anchor_idx]
                if negatives:
                    triplets.append((anchor_idx, pos_idx, negatives[0]))
    return triplets


def train_model(model, features_tensor, loss_fn, optimizer, device, epochs=15,
                eval_fn=None, eval_data=None, eval_every=10, early_stopping_enabled=True,
                actor_idx_bag=None, actor_offsets=None,
                director_idx_bag=None, director_offsets=None,
                genre_idx_bag=None, genre_offsets=None,
                batch_size=512,train_user_item_dict=None):

    features_tensor = features_tensor.to(device)
    actor_idx_bag = actor_idx_bag.to(device)
    actor_offsets = actor_offsets.to(device)
    director_idx_bag = director_idx_bag.to(device)
    director_offsets = director_offsets.to(device)
    genre_idx_bag = genre_idx_bag.to(device)
    genre_offsets = genre_offsets.to(device)
    model.to(device)

    early_stopper = EarlyStopping(patience=3)
    best_ndcg = 0

    for epoch in range(1, epochs + 1):
        print(f"\n Epoch {epoch}")

        model.eval()
        with torch.no_grad():
            item_embeddings = model.predict_embeddings(
                features_tensor,
                actor_idx_bag, actor_offsets,
                director_idx_bag, director_offsets,
                genre_idx_bag, genre_offsets
            )

        triplets = generate_hard_triplets_faiss(
            embeddings=item_embeddings,
            user_item_dict=train_user_item_dict,
            movie_id_map=eval_data["movie_id_map"]
        )
        random.shuffle(triplets)

        model.train()
        total_loss = 0.0
        loop = tqdm(range(0, len(triplets), batch_size), desc=f" Training Epoch {epoch}")

        for i in loop:
            batch = triplets[i:i + batch_size]
            anchor_ids = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_ids = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_ids = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            anc_bag, anc_offs = get_embedding_bag_inputs(anchor_ids, actor_idx_bag, actor_offsets)
            pos_bag, pos_offs = get_embedding_bag_inputs(pos_ids, actor_idx_bag, actor_offsets)
            neg_bag, neg_offs = get_embedding_bag_inputs(neg_ids, actor_idx_bag, actor_offsets)

            dir_anc_bag, dir_anc_offs = get_embedding_bag_inputs(anchor_ids, director_idx_bag, director_offsets)
            dir_pos_bag, dir_pos_offs = get_embedding_bag_inputs(pos_ids, director_idx_bag, director_offsets)
            dir_neg_bag, dir_neg_offs = get_embedding_bag_inputs(neg_ids, director_idx_bag, director_offsets)

            gen_anc_bag, gen_anc_offs = get_embedding_bag_inputs(anchor_ids, genre_idx_bag, genre_offsets)
            gen_pos_bag, gen_pos_offs = get_embedding_bag_inputs(pos_ids, genre_idx_bag, genre_offsets)
            gen_neg_bag, gen_neg_offs = get_embedding_bag_inputs(neg_ids, genre_idx_bag, genre_offsets)

            anchor_vec = model(features_tensor[anchor_ids], anc_bag.to(device), anc_offs.to(device),
                                dir_anc_bag.to(device), dir_anc_offs.to(device),
                                gen_anc_bag.to(device), gen_anc_offs.to(device))

            pos_vec = model(features_tensor[pos_ids], pos_bag.to(device), pos_offs.to(device),
                            dir_pos_bag.to(device), dir_pos_offs.to(device),
                            gen_pos_bag.to(device), gen_pos_offs.to(device))

            neg_vec = model(features_tensor[neg_ids], neg_bag.to(device), neg_offs.to(device),
                            dir_neg_bag.to(device), dir_neg_offs.to(device),
                            gen_neg_bag.to(device), gen_neg_offs.to(device))

            loss = loss_fn(anchor_vec, pos_vec, neg_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / ((i // batch_size) + 1))

        print(f"✅ Avg Loss = {total_loss / max(1, len(triplets) // batch_size):.6f}")

        if eval_fn and eval_data and epoch % eval_every == 0:
            print("🧪 Ewaluacja...")
            model.eval()
            with torch.no_grad():
                item_embeddings = model.predict_embeddings(
                    features_tensor,
                    actor_idx_bag, actor_offsets,
                    director_idx_bag, director_offsets,
                    genre_idx_bag, genre_offsets
                ).cpu().numpy()

                # ✅ ZAPISZ embeddingi do pliku .npy
                os.makedirs("embedding_logs", exist_ok=True)
                np.save(f"embedding_logs/embeddings_epoch_{epoch:02}.npy", item_embeddings)

                # ✅ Zapisz metadata tylko raz
                if epoch == 1:
                    idx_to_movieId = {v: k for k, v in eval_data["movie_id_map"].items()}
                    rows = []
                    for idx in range(item_embeddings.shape[0]):
                        movie_id = idx_to_movieId.get(idx)
                        title = eval_data.get("movie_id_to_title", {}).get(movie_id, "Unknown")
                        rows.append({
                            "index": idx,
                            "movieId": movie_id,
                            "title": title
                        })

                    import pandas as pd
                    pd.DataFrame(rows).to_csv("embedding_logs/embedding_metadata.csv", index=False)

                # 🔄 Potem ewaluacja metryk
                metrics = eval_fn(**eval_data, item_embeddings=item_embeddings)

                for group_name in ['warm_start', 'cold_start']:
                    print(f"📊 {group_name.upper()}:")
                    for metric, value in metrics[group_name].items():
                        print(f"  {metric}: {value:.4f}")

                print("\n🔎 Przykładowe rekomendacje (Top@10 + Ground Truth z tytułami):")

                rev_map = eval_data.get("reverse_movie_id_map", {})
                title_map = eval_data.get("movie_id_to_title", {})

                for uid, recs in list(metrics["examples"].items())[:2]:
                    print(f"\nUser {uid}:")

                    top10 = []
                    for idx, score in recs:
                        movie_id = rev_map.get(idx, None)
                        title = title_map.get(movie_id, "⛔ brak tytułu")
                        top10.append(f"{movie_id} | {title} ({score:.3f})")
                    print("Top@10:")
                    for line in top10:
                        print(f"  - {line}")

                    true_ids = eval_data["test_user_item_dict"].get(uid, [])
                    true_lines = []
                    for mid in true_ids:
                        title = title_map.get(mid, "⛔ brak tytułu")
                        true_lines.append(f"{mid} | {title}")
                    print("Ground truth:")
                    for line in true_lines:
                        print(f"  ✔ {line}")

                current_ndcg = metrics.get("nDCG@K", 0.0)
                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    torch.save(model.state_dict(), "best_item_tower_model.pt")
                    print(f"💾 Zapisano nowy najlepszy model (nDCG@K = {current_ndcg:.4f})")

                if early_stopping_enabled:
                    early_stopper(1 - current_ndcg)
                    if early_stopper.early_stop:
                        print("⏹️ Early stopping: nDCG@K się nie poprawia.")
                        plot_embedding_distribution(item_embeddings)
                        break

    return model