import pandas as pd
import numpy as np
import copy
import random
import time
from collections import defaultdict
from tqdm import tqdm
from surprise import Dataset, Reader, SVD
from RatingLib import User, Movie
import test_users

from RatingSystem import RatingSystem

class SVDSystem(RatingSystem):
    def __init__(self, n_factors=20, n_epochs=20, max_train_samples=2500000):
        super().__init__()
                
        test_set = {tuple(pair) for pair in test_users.test_pairs}
        test_u = {int(p[0]) for p in test_users.test_pairs}
        test_m = {int(p[1]) for p in test_users.test_pairs}
        
        train_data = []
        for u, user_obj in self.users.items():
            for m, r in user_obj.ratings.items():
                if (u, m) not in test_set:
                    # Dodajemy pary (użytkownik, film), które są w zbiorze testowym (ale bez samej oceny testowej)
                    # oraz próbkę pozostałych danych dla szybkości uczenia.
                    if u in test_u or m in test_m:
                        train_data.append({'userID': u, 'itemID': m, 'rating': r})
                    elif random.random() < 0.05:
                        train_data.append({'userID': u, 'itemID': m, 'rating': r})
                        
        # Ograniczenie liczby próbek, aby trening nie trwał zbyt długo
        if len(train_data) > max_train_samples:
            random.shuffle(train_data)
            train_data = train_data[:max_train_samples]
                    
        df = pd.DataFrame(train_data)
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        
        trainset = data.build_full_trainset()
        
        # Inicjalizacja algorytmu SVD (Matrix Factorization) z biasami
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=0.005, reg_all=0.05)
        self.algo.fit(trainset)

    def rate(self, user, movie):
        pred = self.algo.predict(user.id, movie).est
        return pred

    def __str__(self):
        return 'Surprise SVD'

class GenreSystem(RatingSystem):
    # System rekomendacyjny oparty na gatunkach (Genre Selection).
    # Jest to system typu Content-Based / Hybrid, wykorzystujący biasy użytkowników, filmów i gatunków.
    def __init__(self, n_epochs=12, lr=0.015, reg=0.05, max_train_samples=2500000):
        super().__init__()
        
        start_time = time.time()
        
        self.global_mean = 0.0
        
        test_set = {tuple(pair) for pair in test_users.test_pairs}
        test_u = {int(p[0]) for p in test_users.test_pairs}
        test_m = {int(p[1]) for p in test_users.test_pairs}
        
        raw_train_data = []
        total_sum = 0
        total_count = 0
        
        # Obliczamy średnią na danych nietestowych.
        for u, user_obj in self.users.items():
            for m, r in user_obj.ratings.items():
                if (u, m) not in test_set:
                    total_sum += r
                    total_count += 1
                    
                    if u in test_u or m in test_m:
                        raw_train_data.append((u, m, r))
                    elif random.random() < 0.05:
                        raw_train_data.append((u, m, r))

        if total_count > 0:
            self.global_mean = total_sum / total_count
        else:
            self.global_mean = 3.5
            
        if len(raw_train_data) > max_train_samples:
            random.shuffle(raw_train_data)
            raw_train_data = raw_train_data[:max_train_samples]
                        
        # Mapowanie ID użytkowników i filmów na indeksy ciągłe (0, 1, 2...)
        self.u_map = {}
        self.m_map = {}
        u_idx = 0
        m_idx = 0
        
        self.movie_catalog = Movie.index
        
        # Zbieranie wszystkich unikalnych gatunków filmowych
        all_genres = set()
        for m_obj in self.movie_catalog.values():
            for g in m_obj.genres:
                all_genres.add(g)
        self.all_genres = list(all_genres)
        self.g_map = {g: i for i, g in enumerate(self.all_genres)}
        g_idx = len(self.all_genres)
        
        # Przygotowanie listy do pętli SGD (pre-lookup)
        optimized_train_data = []
        for u, m, r in raw_train_data:
            if u not in self.u_map:
                self.u_map[u] = u_idx
                u_idx += 1
            if m not in self.m_map:
                self.m_map[m] = m_idx
                m_idx += 1
            
            uid = self.u_map[u]
            mid = self.m_map[m]
            movie_obj = self.movie_catalog.get(m)
            g_ids = [self.g_map[g] for g in movie_obj.genres if g in self.g_map] if movie_obj else []
            optimized_train_data.append((uid, mid, r, g_ids))

        # Inicjalizacja parametrów modelu (biasy)
        self.bu = [0.0] * u_idx # Bias użytkownika (czy ocenia surowo czy łagodnie)
        self.bm = [0.0] * m_idx # Bias filmu (czy film jest ogólnie lubiany czy nie)
        
        # user-genre bias: self.bug[uid][gid] (jak bardzo user UID lubi gatunek GID)
        self.bug = [[0.0] * g_idx for _ in range(u_idx)]
                
        global_mean = self.global_mean
        # Algorytm optymalizacji SGD (Stochastic Gradient Descent)
        for epoch in range(n_epochs):
            ep_start = time.time()
            for uid, mid, r, g_ids in optimized_train_data:
                num_g = len(g_ids)
                
                # Obliczanie aktualnej predykcji na podstawie średniej z preferencji gatunkowych
                bug_sum = 0.0
                if num_g > 0:
                    for gid in g_ids:
                        bug_sum += self.bug[uid][gid]
                    bug_sum /= num_g # Średnia z biasów gatunkowych usera dla tego filmu
                    
                # Model predykcji: średnia + bias_usera + bias_filmu + średni_pref_gatunkowy
                pred = global_mean + self.bu[uid] + self.bm[mid] + bug_sum
                err = r - pred # Błąd predykcji
                
                # Aktualizacja parametrów zgodnie z kierunkiem gradientu błędu
                self.bu[uid] += lr * (err - reg * self.bu[uid])
                self.bm[mid] += lr * (err - reg * self.bm[mid])
                
                # Aktualizujemy bias gatunkowy bezpośrednio błędem
                if num_g > 0:
                    for gid in g_ids:
                        self.bug[uid][gid] += lr * (err - reg * self.bug[uid][gid])
                                
    def rate(self, user, movie):
        pred = self.global_mean
        uid = self.u_map.get(user.id)
        mid = self.m_map.get(movie)
        
        # Jeśli znamy użytkownika, dodajemy jego ogólny bias
        if uid is not None:
            pred += self.bu[uid]
        # Jeśli znamy film, dodajemy jego ogólny bias
        if mid is not None:
            pred += self.bm[mid]
            
        movie_obj = Movie.index.get(movie)
        
        # Obliczanie wkładu gatunkowego (zainteresowania usera gatunkami tego filmu)
        if uid is not None and movie_obj is not None:
            bug_sum = 0.0
            num_g = 0
            for g in movie_obj.genres:
                gid = self.g_map.get(g)
                if gid is not None:
                    bug_sum += self.bug[uid][gid]
                    num_g += 1
            if num_g > 0:
                pred += bug_sum / num_g
            
        # Ograniczenie wyniku do skali 0.5 - 5.0
        if pred > 5.0: pred = 5.0
        if pred < 0.5: pred = 0.5
        
        return pred

    def __str__(self):
        return 'Genre System'

class HybridSystem(RatingSystem):
    def __init__(self):
        self.svd = SVDSystem()
        self.genre = GenreSystem()
        self.alpha = 0.6

    def rate(self, user, movie):
        pred_svd = self.svd.rate(user, movie)
        pred_genre = self.genre.rate(user, movie)
        
        return self.alpha * pred_svd + (1.0 - self.alpha) * pred_genre

    def __str__(self):
        return f'System created by 155987 and 155976'