import os
import csv
from RatingSystem import RatingSystem
import test_users

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        
        self.global_mean = 0
        total_ratings = 0

        for movie_id, ratings in self.movie_ratings.items():
            self.global_mean += sum(ratings)
            total_ratings += len(ratings)
        
        if total_ratings > 0:
            self.global_mean /= total_ratings
        else:
            self.global_mean = 2.5

        self.movie_genres = {}
        filepath = 'data/movie.csv' if os.path.exists('data/movie.csv') else '../data/movie.csv'
        
        try:
            with open(filepath, encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for line in csv_reader:
                    movie_id = int(line[0])
                    genres = line[2].split('|')
                    self.movie_genres[movie_id] = genres
        except Exception as e:
            print(f"Nie udało się wczytać gatunków. Błąd: {e}")

        self.movie_bias = {}
        movie_bias_sum = {}
        movie_bias_count = {}

        test_set = {tuple(pair) for pair in test_users.test_pairs}

        for u_id, user_object in self.users.items():
            safe_ratings = {}
            for movie_id, rating in user_object.ratings.items():
                if (u_id, movie_id) not in test_set: 
                    safe_ratings[movie_id] = rating

            if not safe_ratings: 
                continue

            u_mean = sum(safe_ratings.values()) / len(safe_ratings)
            
            for movie_id, rating in user_object.ratings.items():
                diff = rating - u_mean
                movie_bias_sum[movie_id] = movie_bias_sum.get(movie_id, 0) + diff
                movie_bias_count[movie_id] = movie_bias_count.get(movie_id, 0) + 1

        lambda_reg = 5.0
        for movie_id in movie_bias_sum:
            self.movie_bias[movie_id] = movie_bias_sum[movie_id] / (movie_bias_count[movie_id] + lambda_reg)

    def rate(self, user, movie):
        if not user.ratings:
            user_mean = self.global_mean
        else:
            user_mean = sum(user.ratings.values()) / len(user.ratings)

        m_bias = self.movie_bias.get(movie, 0)
        pred = user_mean + m_bias

        if movie in self.movie_genres and user.ratings:
            target_genres = self.movie_genres[movie]
            genre_diffs = []

            for g in target_genres:
                g_sum = 0
                g_count = 0
                for movie_id, r in user.ratings.items():
                    if movie_id in self.movie_genres and g in self.movie_genres[movie_id]:
                        g_sum += (r - user_mean)
                        g_count += 1
                if g_count > 0:
                    genre_diffs.append(g_sum / (g_count + 2.0))

            if genre_diffs:
                pred += (sum(genre_diffs) / len(genre_diffs)) * 0.5

        if pred > 5.0:
            return 5.0
        elif pred < 0.5:
            return 0.5
        else:
            return pred

    def __str__(self):
        return 'System created by 155937, 155835 and 156014'