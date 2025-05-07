# # Finalnie dostosowana funkcja ewaluacyjna, eliminująca ValueError przy sprawdzaniu list
# import torch
# import numpy as np
#
#
# def precision_at_k(true_item, recommended_items, k):
#     return int(true_item in recommended_items[:k]) / k
#
#
# def recall_at_k(true_item, recommended_items, k):
#     return int(true_item in recommended_items[:k]) / 1
#
#
# def mrr(true_item, recommended_items):
#     if true_item in recommended_items:
#         return 1 / (recommended_items.index(true_item) + 1)
#     return 0
#
#
# def leave_one_out_split(user_item_dict):
#     train_dict = {}
#     test_dict = {}
#     for user, items in user_item_dict.items():
#         if isinstance(items, np.ndarray):
#             items = items.tolist()
#         if len(items) < 2:
#             continue
#         train_dict[user] = items[:-1]
#         test_dict[user] = items[-1]
#     return train_dict, test_dict
#
#
# @torch.no_grad()
# def evaluate_model_embeddings(user_item_dict, item_embeddings, k=10, similarity='dot'):
#     train_dict, test_items = leave_one_out_split(user_item_dict)
#
#     precisions, recalls, mrrs = [], [], []
#
#     item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
#     item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)
#
#     for user_id in test_items:
#         user_history = train_dict.get(user_id, [])
#         if isinstance(user_history, np.ndarray):
#             user_history = user_history.tolist()
#         if len(user_history) == 0:
#             continue
#
#         user_vector = item_embeddings[user_history].mean(dim=0, keepdim=True)
#
#         if similarity == 'dot':
#             scores = torch.matmul(item_embeddings, user_vector.T).squeeze()
#         elif similarity == 'cosine':
#             scores = torch.nn.functional.cosine_similarity(item_embeddings, user_vector)
#         else:
#             raise ValueError("similarity must be 'dot' or 'cosine'")
#
#         scores[user_history] = -1e9
#
#         top_k_indices = torch.topk(scores, k).indices.tolist()
#         true_item = test_items[user_id]
#
#         precisions.append(precision_at_k(true_item, top_k_indices, k))
#         recalls.append(recall_at_k(true_item, top_k_indices, k))
#         mrrs.append(mrr(true_item, top_k_indices))
#
#     return {
#         "Precision@K": np.mean(precisions),
#         "Recall@K": np.mean(recalls),
#         "MRR": np.mean(mrrs)
#     }


# Zaktualizowana wersja evaluate_model_embeddings z ograniczeniem liczby użytkowników do ewaluacji (np. 1000)

# import torch
# import numpy as np
# import random
# import matplotlib
# matplotlib.use('TkAgg')  # lub 'Qt5Agg' jeśli masz Qt
#
#
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# def plot_embedding_distribution(embeddings, sample_size=2000, perplexity=30):
#     if isinstance(embeddings, torch.Tensor):
#         embeddings = embeddings.cpu().numpy()
#
#     if len(embeddings) > sample_size:
#         indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
#         embeddings = embeddings[indices]
#
#     print("🔄 Reducing dimensions with PCA...")
#     pca = PCA(n_components=50)
#     reduced = pca.fit_transform(embeddings)
#
#     print("🔄 Applying t-SNE...")
#     tsne = TSNE(n_components=2, perplexity=perplexity, init='random', learning_rate='auto')
#     tsne_result = tsne.fit_transform(reduced)
#
#     plt.figure(figsize=(8, 6))
#     plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, s=10)
#     plt.title("t-SNE Visualization of Item Embeddings")
#     plt.xlabel("Dimension 1")
#     plt.ylabel("Dimension 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# def precision_at_k(true_item, recommended_items, k):
#     return int(true_item in recommended_items[:k]) / k
#
# def recall_at_k(true_item, recommended_items, k):
#     return int(true_item in recommended_items[:k]) / 1
#
# def mrr(true_item, recommended_items):
#     if true_item in recommended_items:
#         return 1 / (recommended_items.index(true_item) + 1)
#     return 0
#
# def ndcg_at_k(true_item, recommended_items, k):
#     if true_item in recommended_items[:k]:
#         rank = recommended_items.index(true_item)
#         return 1 / np.log2(rank + 2)
#     return 0.0
#
# def leave_one_out_split(user_item_dict):
#     train_dict = {}
#     test_dict = {}
#     for user, items in user_item_dict.items():
#         if isinstance(items, np.ndarray):
#             items = items.tolist()
#         if len(items) < 2:
#             continue
#         train_dict[user] = items[:-1]
#         test_dict[user] = items[-1]
#     return train_dict, test_dict
#
# @torch.no_grad()
# def evaluate_model_embeddings(user_item_dict, item_embeddings, k=10, similarity='dot', max_users=1000):
#     train_dict, test_items = leave_one_out_split(user_item_dict)
#
#     # Próbkuj użytkowników do ewaluacji (jeśli max_users ustawione)
#     if max_users is not None and len(test_items) > max_users:
#         sampled_users = random.sample(list(test_items.keys()), k=max_users)
#         test_items = {u: test_items[u] for u in sampled_users}
#
#     precisions, recalls, mrrs, ndcgs = [], [], [], []
#
#     item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
#     item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)
#
#     for user_id in test_items:
#         user_history = train_dict.get(user_id, [])
#         if isinstance(user_history, np.ndarray):
#             user_history = user_history.tolist()
#         user_history = [i for i in user_history if 0 <= i < item_embeddings.shape[0]]
#
#         if len(user_history) == 0:
#             continue
#
#         user_tensor = item_embeddings[user_history]
#         if user_tensor.numel() == 0:
#             continue
#
#         user_vector = user_tensor.mean(dim=0, keepdim=True)
#
#         if similarity == 'dot':
#             scores = torch.matmul(item_embeddings, user_vector.T).squeeze()
#         elif similarity == 'cosine':
#             scores = torch.nn.functional.cosine_similarity(item_embeddings, user_vector)
#         else:
#             raise ValueError("similarity must be 'dot' or 'cosine'")
#
#         scores[user_history] = -1e9
#         top_k_indices = torch.topk(scores, k).indices.tolist()
#         true_item = test_items[user_id]
#
#         precisions.append(precision_at_k(true_item, top_k_indices, k))
#         recalls.append(recall_at_k(true_item, top_k_indices, k))
#         mrrs.append(mrr(true_item, top_k_indices))
#         ndcgs.append(ndcg_at_k(true_item, top_k_indices, k))
#
#     return {
#         "Precision@K": np.mean(precisions) if precisions else 0.0,
#         "Recall@K": np.mean(recalls) if recalls else 0.0,
#         "MRR": np.mean(mrrs) if mrrs else 0.0,
#         "nDCG@K": np.mean(ndcgs) if ndcgs else 0.0
#     }


# NOWE

# Zaktualizowana wersja evaluate_model_embeddings z ograniczeniem liczby użytkowników do ewaluacji (np. 1000)

import torch
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # lub 'Qt5Agg' jeśli masz Qt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embedding_distribution(embeddings, sample_size=2000, perplexity=30):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]

    print("🔄 Reducing dimensions with PCA...")
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(embeddings)

    print("🔄 Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', learning_rate='auto')
    tsne_result = tsne.fit_transform(reduced)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, s=10)
    plt.title("t-SNE Visualization of Item Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def precision_at_k(true_item, recommended_items, k):
    return int(true_item in recommended_items[:k]) / k

def recall_at_k(true_item, recommended_items, k):
    return int(true_item in recommended_items[:k]) / 1

def mrr(true_item, recommended_items):
    if true_item in recommended_items:
        return 1 / (recommended_items.index(true_item) + 1)
    return 0

def ndcg_at_k(true_item, recommended_items, k):
    if true_item in recommended_items[:k]:
        rank = recommended_items.index(true_item)
        return 1 / np.log2(rank + 2)
    return 0.0

def leave_one_out_split(user_item_dict):
    train_dict = {}
    test_dict = {}
    for user, items in user_item_dict.items():
        if isinstance(items, np.ndarray):
            items = items.tolist()
        if len(items) < 2:
            continue
        train_dict[user] = items[:-1]
        test_dict[user] = items[-1]
    return train_dict, test_dict

@torch.no_grad()
def evaluate_model_embeddings_full(train_user_item_dict,
                                   test_user_item_dict,
                                   item_embeddings,
                                   movie_id_map,
                                   ks=[5, 10, 20],
                                   similarity='dot',
                                   max_users=1000,
                                   reverse_movie_id_map=None,
                                   movie_id_to_title=None):
    from collections import defaultdict

    item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
    item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)

    all_users = list(test_user_item_dict.keys())
    if max_users and len(all_users) > max_users:
        all_users = random.sample(all_users, k=max_users)

    metrics_warm = defaultdict(list)
    metrics_cold = defaultdict(list)
    example_recommendations = {}

    for user_id in all_users:
        test_items = test_user_item_dict[user_id]
        train_items = train_user_item_dict.get(user_id, [])

        # mapowanie do indeksów
        history_indices = [movie_id_map.get(m, -1) for m in train_items]
        history_indices = [i for i in history_indices if 0 <= i < item_embeddings.shape[0]]
        true_indices = [movie_id_map.get(m, -1) for m in test_items if movie_id_map.get(m, -1) is not None]

        if not true_indices:
            continue

        if history_indices:
            user_vector = item_embeddings[history_indices].mean(dim=0, keepdim=True)
            is_cold = False
        else:
            user_vector = torch.randn_like(item_embeddings[0]).unsqueeze(0)
            is_cold = True

        if similarity == 'dot':
            scores = torch.matmul(item_embeddings, user_vector.T).squeeze()
        elif similarity == 'cosine':
            scores = torch.nn.functional.cosine_similarity(item_embeddings, user_vector)
        else:
            raise ValueError("similarity must be 'dot' or 'cosine'")

        for idx in history_indices:
            scores[idx] = -1e9  # wykluczenie znanych filmów

        topk_scores, topk_indices = torch.topk(scores, max(ks))
        topk_indices = topk_indices.tolist()
        topk_scores = topk_scores.tolist()

        # Zachowaj jeden przykład do analizy
        if len(example_recommendations) < 10:
            example_recommendations[user_id] = list(zip(topk_indices[:10], topk_scores[:10]))

        # Oblicz metryki dla każdego k
        for k in ks:
            precision_sum, recall_sum, mrr_sum, ndcg_sum = 0, 0, 0, 0
            for true_idx in true_indices:
                top_k = topk_indices[:k]

                precision_sum += precision_at_k(true_idx, top_k, k)
                recall_sum += recall_at_k(true_idx, top_k, k)
                mrr_sum += mrr(true_idx, top_k)
                ndcg_sum += ndcg_at_k(true_idx, top_k, k)

            num_items = len(true_indices)
            if is_cold:
                metrics_cold[f'Precision@{k}'].append(precision_sum / num_items)
                metrics_cold[f'Recall@{k}'].append(recall_sum / num_items)
                metrics_cold[f'MRR@{k}'].append(mrr_sum / num_items)
                metrics_cold[f'nDCG@{k}'].append(ndcg_sum / num_items)
            else:
                metrics_warm[f'Precision@{k}'].append(precision_sum / num_items)
                metrics_warm[f'Recall@{k}'].append(recall_sum / num_items)
                metrics_warm[f'MRR@{k}'].append(mrr_sum / num_items)
                metrics_warm[f'nDCG@{k}'].append(ndcg_sum / num_items)

    # Agreguj wyniki
    def summarize(metrics):
        return {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}

    # print(f"\n📊 Ewaluacja zakończona:")
    # print(f"Warm-start users: {len(metrics_warm[f'Precision@{ks[0]}'])}")
    # print(f"Cold-start users: {len(metrics_cold[f'Precision@{ks[0]}'])}")

    return {
        "warm_start": summarize(metrics_warm),
        "cold_start": summarize(metrics_cold),
        "examples": example_recommendations
    }



# NOWY k-fold

# import torch
# import numpy as np
# import random
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib
# matplotlib.use('TkAgg')  # lub 'Qt5Agg' jeśli masz Qt
# import matplotlib.pyplot as plt
#
#
# # Funkcja t-SNE dla wizualizacji
# def plot_embedding_distribution(embeddings, sample_size=2000, perplexity=30):
#     if isinstance(embeddings, torch.Tensor):
#         embeddings = embeddings.cpu().numpy()
#
#     if len(embeddings) > sample_size:
#         indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
#         embeddings = embeddings[indices]
#
#     print("🔄 Reducing dimensions with PCA...")
#     pca = PCA(n_components=50)
#     reduced = pca.fit_transform(embeddings)
#
#     print("🔄 Applying t-SNE...")
#     tsne = TSNE(n_components=2, perplexity=perplexity, init='random', learning_rate='auto')
#     tsne_result = tsne.fit_transform(reduced)
#
#     plt.figure(figsize=(8, 6))
#     plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, s=10)
#     plt.title("t-SNE Visualization of Item Embeddings")
#     plt.xlabel("Dimension 1")
#     plt.ylabel("Dimension 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# # Metryki ewaluacji
# def precision_at_k(true_item, recommended_items, k):
#     return int(true_item in recommended_items[:k]) / k
#
#
# def recall_at_k(true_item, recommended_items, k):
#     return int(true_item in recommended_items[:k]) / 1
#
#
# def mrr(true_item, recommended_items):
#     if true_item in recommended_items:
#         return 1 / (recommended_items.index(true_item) + 1)
#     return 0
#
#
# def ndcg_at_k(true_item, recommended_items, k):
#     if true_item in recommended_items[:k]:
#         rank = recommended_items.index(true_item)
#         return 1 / np.log2(rank + 2)
#     return 0.0
#
#
# # Zmodyfikowana funkcja do k-fold cross-validation
# def k_fold_split(user_item_dict, k=5):
#     users = list(user_item_dict.keys())
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#
#     folds = []
#     for train_idx, test_idx in kf.split(users):
#         train_users = [users[i] for i in train_idx]
#         test_users = [users[i] for i in test_idx]
#
#         train_dict = {user: user_item_dict[user] for user in train_users}
#         test_dict = {user: user_item_dict[user] for user in test_users}
#
#         folds.append((train_dict, test_dict))
#
#     return folds
#
#
# @torch.no_grad()
# def evaluate_model_embeddings(user_item_dict, item_embeddings, movie_id_map, k=10, similarity='dot', max_users=1000, **kwargs):
#     # Tutaj już nie musisz wywoływać k_fold_split, ponieważ foldy są przekazane
#     precisions, recalls, mrrs, ndcgs = [], [], [], []
#
#     item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
#     item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)
#
#     # Ewaluacja na podstawie przekazanych danych
#     for user_id in user_item_dict:
#         user_history_ids = user_item_dict.get(user_id, [])
#         true_item_id = user_item_dict[user_id]  # Sprawdź, czy test_items[user_id] nie jest listą
#         if isinstance(true_item_id, list):
#             true_item_id = true_item_id[0]  # Wybierz pierwszy element, jeśli to lista
#
#         # Mapowanie do indeksów
#         user_history = [movie_id_map.get(m_id, -1) for m_id in user_history_ids]
#         user_history = [i for i in user_history if 0 <= i < item_embeddings.shape[0]]
#         true_item = movie_id_map.get(true_item_id, -1)
#
#         if true_item == -1 or len(user_history) == 0:
#             continue
#
#         user_tensor = item_embeddings[user_history]
#         if user_tensor.numel() == 0:
#             continue
#
#         user_vector = user_tensor.mean(dim=0, keepdim=True)
#
#         if similarity == 'dot':
#             scores = torch.matmul(item_embeddings, user_vector.T).squeeze()
#         elif similarity == 'cosine':
#             scores = torch.nn.functional.cosine_similarity(item_embeddings, user_vector)
#         else:
#             raise ValueError("similarity must be 'dot' or 'cosine'")
#
#         for idx in user_history:
#             scores[idx] = -1e9
#
#         top_k_indices = torch.topk(scores, k).indices.tolist()
#
#         precisions.append(precision_at_k(true_item, top_k_indices, k))
#         recalls.append(recall_at_k(true_item, top_k_indices, k))
#         mrrs.append(mrr(true_item, top_k_indices))
#         ndcgs.append(ndcg_at_k(true_item, top_k_indices, k))
#
#     return {
#         "Precision@K": np.mean(precisions),
#         "Recall@K": np.mean(recalls),
#         "MRR": np.mean(mrrs),
#         "nDCG@K": np.mean(ndcgs)
#     }
#


