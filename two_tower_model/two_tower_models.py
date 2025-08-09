import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 64

class UserTower(nn.Module):
    def __init__(self, input_dim, n_items, embedding_dim=EMB_DIM):
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
        x = torch.stack([batch['ratings'], batch['times']], dim=-1) # [B, L_u, 2]
        w = torch.sigmoid(self.rating_proj(x))

        # weighted mean-pool
        pooled = (m * w).sum(1) / w.sum(1).clamp_min(1e-6)   # [B, D]

        input = torch.cat([batch['user_statistics'], pooled], dim=-1) # [B, stats+EMB_DIM]
        output = self.mlp(input)                                    # [B, EMB_DIM]
        u = F.normalize(output, dim = 1)
        return u


class ItemTower(nn.Module):
    def __init__(self,dense_feat_dim,text_emb_dim,vocab_sizes,embedding_dim=EMB_DIM):
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

        self.text_mlp = nn.Sequential( #--- to consider za ostre zejscie z 512 -> 64, moze posredni 256
            nn.Linear(text_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

        MLP_INPUT_DIM = embedding_dim*5 # odpowiednio nn.Embeedings * 3 oraz meta_mlp oraz text_mlp
        self.final_mlp = nn.Sequential(
            nn.Linear(MLP_INPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,embedding_dim)
        )

    def forward(self, batch, key: str = "pos_item"):

        dense_feats = batch[key]['dense_features']     # [B, dense_feat_dim]
        text_emb = batch[key]['text_embedding']     # [B, text_emb_dim]

        actor_ids = batch[key]['actor_ids']         # [B, max_len_a]
        director_ids = batch[key]['director_ids']
        genre_ids = batch[key]['genre_ids']

        if dense_feats.dim() == 3:
            B, k, Z = dense_feats.shape

            # flattenujemy
            dense_flat     = dense_feats.view(B*k, Z)
            text_flat      = text_emb.view(B*k, -1)
            actor_flat     = actor_ids.view(B*k, -1)
            director_flat  = director_ids.view(B*k, -1)
            genre_flat     = genre_ids.view(B*k, -1)

            # złożony batch
            flat_batch = {
                key: {
                    'dense_features':  dense_flat,
                    'text_embedding':  text_flat,
                    'actor_ids':       actor_flat,
                    'director_ids':    director_flat,
                    'genre_ids':       genre_flat,
                }
            }

            emb_flat = self.forward(flat_batch, key)    # rekurencyjnie batch na embeddingi [B*k, D]

            return emb_flat.view(B, k, -1)              # [B, k, D]

        dense_vec = self.meta_mlp(dense_feats)      # [B, D]
        text_vec = self.text_mlp(text_emb)          # [B, D]

        cast_imp = dense_feats[:, 2:3]              # [B, 1]
        director_score = dense_feats[:, 3:4]        # [B, 1]

        a = self.actor_emb   (actor_ids).mean(dim=1)    # [B, D]
        d = self.director_emb(director_ids).mean(dim=1) # [B, D]
        g = self.genre_emb   (genre_ids).mean(dim=1)    # [B, D]

        # We add weights based on importance score
        a = a * cast_imp
        d = d * director_score #--- do rozwazenia Max pooling lub Attention pooling

        input = torch.cat([a, d, g, dense_vec, text_vec], dim=-1)   # [B, 5D]
        output = self.final_mlp(input)                              # [B, D]
        i = F.normalize(output, dim=1)
        return i
    
class TwoTowerModel(nn.Module):
    def __init__(self, stats_dim, n_items, vocab_sizes,
                 dense_feat_dim, text_emb_dim, embedding_dim=EMB_DIM):
        super().__init__()
        self.user_tower = UserTower(stats_dim, n_items, embedding_dim)
        self.item_tower = ItemTower(dense_feat_dim, text_emb_dim, vocab_sizes, embedding_dim)

    def forward(self, batch):
        u = self.user_tower(batch['user'])
        i_pos = self.item_tower(batch, key="pos_item")
        i_neg = self.item_tower(batch, key="neg_item")

        if i_neg.dim() == 2:
            return u, i_pos, i_neg # każdy [B, 64]

        B, k, D = i_neg.shape

        i_neg_flat = i_neg.reshape(B*k, D) # Splaszczamy

        u_flat = u.unsqueeze(1).expand(B, k, D).reshape(B*k, D)
        pos_flat = i_pos.unsqueeze(1).expand(B, k, D).reshape(B*k, D)

        return u_flat, pos_flat, i_neg_flat
