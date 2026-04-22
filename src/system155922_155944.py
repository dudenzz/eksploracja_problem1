import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import pandas as pd
import numpy as np
import csv
from RatingSystem import RatingSystem
import test_users

CONFIG = {
    'movies_path': '../data/movie.csv',
    'ratings_path': '../data/rating.csv',
    'batch_size': 4096,
    'embedding_dim': 64,
    'epochs': 2,
    'lr': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'train_split': 0.9
}


def create_mappings(ratings_path, movies_path):
    user_set, movie_set = set(), set()
    for chunk in pd.read_csv(movies_path, usecols=['movieId'], chunksize=100000):
        movie_set.update(chunk['movieId'].unique())
    for chunk in pd.read_csv(ratings_path, usecols=['userId', 'movieId'], chunksize=500000):
        user_set.update(chunk['userId'].unique())
        movie_set.update(chunk['movieId'].unique())
    user2idx = {id: i for i, id in enumerate(sorted(user_set))}
    movie2idx = {id: i for i, id in enumerate(sorted(movie_set))}
    with open('user2idx.pkl', 'wb') as f:
        pickle.dump(user2idx, f)
    with open('movie2idx.pkl', 'wb') as f:
        pickle.dump(movie2idx, f)
    return user2idx, movie2idx


class RatingsDataset(IterableDataset):
    def __init__(self, file_path, user2idx, movie2idx, chunksize=10000, mode='train', split_ratio=0.9):
        self.file_path = file_path
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.chunksize = chunksize
        self.mode = mode
        self.split_ratio = split_ratio

    def __iter__(self):
        worker_info = get_worker_info()
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunksize):
            u_idx = chunk['userId'].map(self.user2idx).values
            m_idx = chunk['movieId'].map(self.movie2idx).values
            ratings = ((chunk['rating'].values - 1) / 4).astype(np.float32)
            n = len(chunk)
            split_idx = int(n * self.split_ratio)
            start, end = (0, split_idx) if self.mode == 'train' else (split_idx, n)
            if worker_info:
                per_worker = (end - start) // worker_info.num_workers
                w_start = start + worker_info.id * per_worker
                w_end = w_start + per_worker if worker_info.id != worker_info.num_workers - 1 else end
            else:
                w_start, w_end = start, end
            for i in range(w_start, w_end):
                yield torch.tensor(u_idx[i], dtype=torch.long), torch.tensor(m_idx[i], dtype=torch.long), torch.tensor(ratings[i], dtype=torch.float32)


class EnhancedRecommender(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=64):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, emb_dim)
        self.m_emb = nn.Embedding(n_movies, emb_dim)
        self.u_bias = nn.Embedding(n_users, 1)
        self.m_bias = nn.Embedding(n_movies, 1)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.m_emb.weight)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.m_bias.weight)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, u_idx, m_idx):
        u = self.u_emb(u_idx)
        m = self.m_emb(m_idx)
        interaction = u * m
        x = torch.cat([u, m, interaction], dim=1)
        out = self.mlp(x).squeeze(-1) + self.u_bias(u_idx).squeeze(-1) + self.m_bias(m_idx).squeeze(-1)
        return self.sigmoid(out)


class MySystem(RatingSystem):
    _model = None
    _user2idx = None
    _movie2idx = None
    _is_ready = False

    def __init__(self):
        super().__init__()
        self._ensure_ready()

    @classmethod
    def _ensure_ready(cls):
        if cls._is_ready:
            return
        if not (os.path.exists('../model/model.pth') and os.path.exists('../model/user2idx.pkl') and os.path.exists('../model/movie2idx.pkl')):
            print('Brak wytrenowanego modelu, rozpoczynam trening...')
            cls._train()
        cls._load_model()
        cls._is_ready = True

    @classmethod
    def _train(cls):
        # Filtruj rating.csv, usuwając wiersze z test_pairs
        test_set = set(tuple(pair) for pair in test_users.test_pairs)
        filtered_ratings_path = '../data/rating_filtered.csv'
        with open(CONFIG['ratings_path'], 'r', encoding='utf-8') as f_in, open(filtered_ratings_path, 'w+', encoding='utf-8') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                user_id, movie_id = int(row[0]), int(row[1])
                if (user_id, movie_id) not in test_set:
                    writer.writerow(row)
        original_path = CONFIG['ratings_path']
        CONFIG['ratings_path'] = filtered_ratings_path
        user2idx, movie2idx = create_mappings(CONFIG['ratings_path'], CONFIG['movies_path'])
        train_ds = RatingsDataset(CONFIG['ratings_path'], user2idx, movie2idx, mode='train')
        val_ds = RatingsDataset(CONFIG['ratings_path'], user2idx, movie2idx, mode='val')
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=4, pin_memory=True)
        model = EnhancedRecommender(len(user2idx), len(movie2idx), CONFIG['embedding_dim']).to(CONFIG['device'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        for epoch in range(CONFIG['epochs']):
            model.train()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoka {epoch + 1}/{CONFIG['epochs']} | Current LR: {current_lr}")
            for u, m, r in train_loader:
                u, m, r = u.to(CONFIG['device']), m.to(CONFIG['device']), r.to(CONFIG['device'])
                optimizer.zero_grad()
                loss = criterion(model(u, m), r)
                loss.backward()
                optimizer.step()
            model.eval()
            val_mse, samples = 0.0, 0
            with torch.no_grad():
                for u, m, r in val_loader:
                    u, m, r = u.to(CONFIG['device']), m.to(CONFIG['device']), r.to(CONFIG['device'])
                    val_mse += nn.functional.mse_loss(model(u, m), r, reduction='sum').item()
                    samples += r.size(0)
            avg_mse = val_mse / samples if samples > 0 else 0.0
            rmse = np.sqrt(avg_mse) if samples > 0 else 0.0
            print(f"Validation MSE: {avg_mse:.4f} | RMSE: {rmse:.4f}")
            scheduler.step(avg_mse)
        torch.save(model.state_dict(), 'model.pth')
        torch.save(model.u_emb.weight.data, 'user_embeddings.pt')
        torch.save(model.m_emb.weight.data, 'movie_embeddings.pt')
        cls._user2idx = user2idx
        cls._movie2idx = movie2idx
        cls._model = model.eval()
        CONFIG['ratings_path'] = original_path

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return
        with open('../model/user2idx.pkl', 'rb') as f:
            cls._user2idx = pickle.load(f)
        with open('../model/movie2idx.pkl', 'rb') as f:
            cls._movie2idx = pickle.load(f)
        model = EnhancedRecommender(len(cls._user2idx), len(cls._movie2idx), CONFIG['embedding_dim'])
        model.load_state_dict(torch.load('../model/model.pth', map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        cls._model = model

    def rate(self, user, movie_id):
        if movie_id in user.ratings:
            return user.ratings[movie_id]
        self._ensure_ready()
        u_idx = self._user2idx.get(user.id)
        m_idx = self._movie2idx.get(movie_id)
        if u_idx is None or m_idx is None:
            return 2.5
        with torch.no_grad():
            u_t = torch.tensor([u_idx], dtype=torch.long, device=CONFIG['device'])
            m_t = torch.tensor([m_idx], dtype=torch.long, device=CONFIG['device'])
            out = self._model(u_t, m_t).item()
        return float(out * 4 + 1)

    def __str__(self):
        return 'System created by 155922 and 155944'


if __name__ == '__main__':
    MySystem()
