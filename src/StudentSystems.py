from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie
import tqdm
import csv
from surprise import SVD
from surprise import Dataset, Reader
import pandas as pd
import math
from collections import defaultdict
from tqdm import tqdm
class Sys147715(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        user_ratings = np.array(list(user.ratings.values()))
        movie_ratings = np.array(self.movie_ratings.get(movie, []))

        user_weight = 0.5
        movie_weight = 0.5

        if user_ratings.size > 0:
            user_mean = np.mean(user_ratings)
            user_median = np.median(user_ratings)
            user_avg = user_weight * user_mean + (1 - user_weight) * user_median
        else:
            user_avg = 2.5

        if movie_ratings.size > 0:
            movie_mean = np.mean(movie_ratings)
            movie_median = np.median(movie_ratings)
            movie_avg = movie_weight * movie_mean + (1 - movie_weight) * movie_median
        else:
            movie_avg = 2.5

        return (user_avg + movie_avg) / 2

    def __str__(self):
        return 'System created by 147715'
    
class Sys151061(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
       

       
        movie_ratings = self.movie_ratings.get(movie, [])
        if movie_ratings:
            median_movie = float(np.median(movie_ratings))
        else:
            median_movie = 2.5

    
        user_ratings = list(user.ratings.values())
        if user_ratings:
            median_user = float(np.median(user_ratings))
        else:
            median_user = 2.5

       
        return (median_movie + median_user) / 2

    def __str__(self):
        return 'System created by 151061'
    
class Sys151118(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        user_ratings = np.array(list(user.ratings.values()))
        movie_ratings = np.array(self.movie_ratings.get(movie, []))
        global_avg = 2.5
        if user_ratings.size > 0:
            user_avg = np.mean(user_ratings)
        else:
            user_avg = global_avg
        if movie_ratings.size > 0:
            movie_avg = np.mean(movie_ratings)
        else:
            movie_avg = global_avg
        return (user_avg + movie_avg) / 2

    def __str__(self):
        return 'System created by 151118'
    
class Sys151129(RatingSystem):
    def __init__(self):
        super().__init__()
        self.global_avg = self._compute_global_avg()

    def _compute_global_avg(self):
        all_ratings = []
        for ratings in self.movie_ratings.values():
            all_ratings.extend(ratings)
        if all_ratings:
            return np.mean(all_ratings)
        else:
            return 2.5

    def _bayesian_avg(self, ratings, global_avg, C=5):
        n = ratings.size
        if n == 0:
            return global_avg
        else:
            return (C * global_avg + ratings.sum()) / (C + n)

    def rate(self, user, movie):

        global_avg = self.global_avg

        user_ratings = np.array(list(user.ratings.values()))
        user_avg = self._bayesian_avg(user_ratings, global_avg) if user_ratings.size > 0 else global_avg

        movie_ratings = np.array(self.movie_ratings.get(movie, []))
        movie_avg = self._bayesian_avg(movie_ratings, global_avg) if movie_ratings.size > 0 else global_avg

        prediction = (user_avg + movie_avg) / 2

        min_rating, max_rating = 0, 5
        return min(max(prediction, min_rating), max_rating)

    def __str__(self):
        return 'System created by 151129'
    
class Sys151481(RatingSystem):
    def __init__(self):
        super().__init__()
        self._prepare_global()
        self._prepare_biases()
        self._prepare_genres()
        self._prepare_user_preferences()

    def _prepare_global(self):
        oceny = [r for rs in self.movie_ratings.values() for r in rs]
        self.global_avg = np.mean(oceny)

    def _prepare_biases(self):
        self.user_bias = {}
        self.movie_bias = {}
        for uid, user in self.users.items():
            user_scores = list(user.ratings.values())
            reg = 5 + 0.1 * len(user_scores)
            self.user_bias[uid] = (np.mean(user_scores) - self.global_avg) * len(user_scores) / (len(user_scores) + reg) if user_scores else 0.0

        for mid, ratings in self.movie_ratings.items():
            reg = 5 + 0.1 * len(ratings)
            self.movie_bias[mid] = (np.mean(ratings) - self.global_avg) * len(ratings) / (len(ratings) + reg) if ratings else 0.0

    def _prepare_genres(self):
        self.movie_genres = {}
        with open('../data/movie.csv', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                mid = int(row[0])
                genres = row[2].split('|') if row[2] else []
                self.movie_genres[mid] = genres

    def _prepare_user_preferences(self):
        self.user_genre_avg = {}
        self.user_genre_cnt = {}
        for uid, user in self.users.items():
            genre_scores = {}
            genre_counts = {}
            for mid, rating in user.ratings.items():
                for genre in self.movie_genres.get(mid, []):
                    genre_scores.setdefault(genre, []).append(rating)
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            self.user_genre_avg[uid] = {g: np.mean(vals) for g, vals in genre_scores.items()}
            self.user_genre_cnt[uid] = genre_counts

    def rate(self, user, movie):
        uid = user.id
        genres = self.movie_genres.get(movie, [])
        bu = self.user_bias.get(uid, 0.0)
        bi = self.movie_bias.get(movie, 0.0)

        g_vals = []
        g_wagi = []
        for g in genres:
            sr = self.user_genre_avg.get(uid, {}).get(g)
            ile = self.user_genre_cnt.get(uid, {}).get(g, 0)
            if sr is not None:
                g_vals.append(sr)
                g_wagi.append(ile)

        if g_vals and g_wagi:
            genre_part = np.average(g_vals, weights=g_wagi)
        elif g_vals:
            genre_part = np.mean(g_vals)
        else:
            genre_part = self.global_avg

        u_conf = len(user.ratings) / (len(user.ratings) + 10)
        m_conf = len(self.movie_ratings.get(movie, [])) / (len(self.movie_ratings.get(movie, [])) + 10)
        conf = (u_conf + m_conf) / 2

        base = 0.72 * (self.global_avg + bu + bi) + 0.28 * genre_part
        final_score = conf * base + (1 - conf) * self.global_avg

        return min(max(final_score, 0.5), 5.0)

    def __str__(self):
        return '151481'
    
class Sys151504(RatingSystem):
    def __init__(self):
        super().__init__()
        self.global_average = self.compute_global_average()
        self.lambda_user = 5
        self.lambda_movie = 10

    def compute_global_average(self):
        total = 0
        count = 0
        for ratings in self.movie_ratings.values():
            total += sum(ratings)
            count += len(ratings)
        return total / count if count != 0 else 2.5
    
    def regularized_average(self, values, regularization_lambda):
        n = len(values)
        if n == 0:
            return self.global_average
        return (sum(values) + regularization_lambda * self.global_average) / (n + regularization_lambda)
            

    def rate(self, user, movie):
        user_ratings = list(user.ratings.values())
        avg_user = self.regularized_average(user_ratings, self.lambda_user)

        movie_ratings = self.movie_ratings.get(movie, [])
        avg_movie = self.regularized_average(movie_ratings, self.lambda_movie) 

        bias_user = avg_user - self.global_average
        bias_movie = avg_movie - self.global_average
        num_user_ratings = len(user_ratings)
    
        
        if user_ratings:
            avg_user = sum(user_ratings) / len(user_ratings)
        else:
            avg_user = self.global_average


        num_movie_ratings = len(movie_ratings)
        if movie_ratings:
            avg_movie = sum(movie_ratings) / len(movie_ratings)
        else:
            avg_movie = self.global_average

        if num_movie_ratings < 2 and num_user_ratings >= 20:
            return avg_user

        bias = 10
        total_ratings = num_user_ratings + num_movie_ratings + bias 
        if total_ratings == 0:
            return self.global_average
        weight_user = num_user_ratings / (num_user_ratings + bias)
        weight_movie = num_movie_ratings / (num_movie_ratings + bias)
        
        
        weight_sum = weight_user + weight_movie
        weight_user /= weight_sum
        weight_movie /= weight_sum
        predicted_rating = self.global_average + bias_user * weight_user + bias_movie * weight_movie
        return max(1.0, min(5.0, predicted_rating))

    def __str__(self):
        return 'System created by 151504'
    
class Sys151561(RatingSystem):
    def __init__(self):
        super().__init__()

        self.movie_genres = {}
        self._load_movie_genres()
        self.user_genre_preferences = self._build_user_genre_profiles()
        self.global_mean_user_review = self._compute_global_user_mean()

    def _load_movie_genres(self):
        with open('../data/movie.csv', encoding='utf-8') as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for line in csvreader:
                movie_id = int(line[0])
                genres = line[2].split('|') if line[2] else []
                self.movie_genres[movie_id] = genres

    def _build_user_genre_profiles(self):
        user_genres = {}
        for user_id, user in self.users.items():
            genre_scores = {}
            genre_counts = {}
            for movie_id, rating in user.ratings.items():
                genres = self.movie_genres.get(movie_id, [])
                for genre in genres:
                    genre_scores[genre] = genre_scores.get(genre, 0) + rating
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            #srednie ocen uzytkownika dla poszczegolnych gatunkow
            user_genres[user_id] = {genre: genre_scores[genre] / genre_counts[genre]
                                    for genre in genre_scores}
        return user_genres

    def _compute_global_user_mean(self):
        total = 0
        count = 0
        for user in self.users.values():
            if user.ratings:
                total += np.mean(list(user.ratings.values()))
                count += 1
        return total / count if count > 0 else 2.5

    def rate(self, user, movie_id):

        #srednie ocen filmu
        movie_ratings = self.movie_ratings.get(movie_id, [])
        movie_mean = np.mean(movie_ratings) if movie_ratings else None

        #medaiana ocen uzytkownika
        user_ratings = list(user.ratings.values())
        user_median = np.median(user_ratings) if user_ratings else None

        #preferencje gatunkowe
        genre_ratings = []
        for genre in self.movie_genres.get(movie_id, []):
            if user.id in self.user_genre_preferences and genre in self.user_genre_preferences[user.id]:
                genre_ratings.append(self.user_genre_preferences[user.id][genre])
        genre_mean = np.mean(genre_ratings) if genre_ratings else None

        #odchylenie od średniej globalnej
        user_mean = np.mean(user_ratings) if user_ratings else None
        user_bias = user_mean - self.global_mean_user_review if user_mean is not None else 0

        movie_mean = movie_mean if movie_mean is not None else self.global_mean_user_review
        user_median = user_median if user_median is not None else self.global_mean_user_review
        genre_mean = genre_mean if genre_mean is not None else self.global_mean_user_review

        prediction = (0.35 * genre_mean + 0.25 * movie_mean + 0.25 * user_median + 0.15 * (movie_mean + user_bias))

        return prediction

    def __str__(self):
        return 'System created by 151561'
    
class Sys151739(RatingSystem):
    def __init__(self):
        
        super().__init__()
        reg_lambda=20
        
        total_sum = 0.0
        total_count = 0
        for ratings in self.movie_ratings.values():
            total_sum += sum(ratings)
            total_count += len(ratings)
        self.global_avg = total_sum / total_count if total_count > 0 else 2.5

        self.movie_bias = {}
        for movie, ratings in self.movie_ratings.items():
            self.movie_bias[movie] = (sum(ratings) / (len(ratings) + reg_lambda)) - self.global_avg
        
        self.user_bias = {}
        for user_id, user in self.users.items():
            user_ratings = list(user.ratings.values())
            if user_ratings:
                self.user_bias[user_id] = (sum(user_ratings) / len(user_ratings)) - self.global_avg
            else:
                self.user_bias[user_id] = 0.0

    def rate(self, user, movie):
        mb = self.movie_bias.get(movie, 0)
        ub = self.user_bias.get(user.id, 0)
        pred = self.global_avg + mb + ub
        
        return max(0, min(5, pred))

    def __str__(self):
        return 'System created by 151739'

class Sys151754(RatingSystem):
    def __init__(self):
        super().__init__()
        self.GlobalAverageMovieRating = 0
        self.TotalMovies = 0
        for movie in self.movie_ratings:
            for rating  in self.movie_ratings[movie]:
                self.GlobalAverageMovieRating += rating
                self.TotalMovies += 1
        self.GlobalAverageMovieRating /= self.TotalMovies
        
    def rate(self, user, movie):
        n = len(self.movie_ratings[movie])
        if n == 0:
            movie_avg =  2.5
        else:
            movie_avg = sum(self.movie_ratings[movie])/n
        
        n = len(user.ratings.values())
        if n == 0:
            user_avg =  2.5
        else:
            user_avg = sum(user.ratings.values())/n
        
        
        return self.GlobalAverageMovieRating + (user_avg - self.GlobalAverageMovieRating) + (movie_avg - self.GlobalAverageMovieRating)
        
    
    def __str__(self):
        return 'System created by 151754'


class Sys151756(RatingSystem):
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.model = None
        self.prepare_data()
        self.train_model()

    def prepare_data(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = []

        for user_id, user in tqdm(self.users.items(), desc="Preparing data", unit="user"):
            for movie_id, rating in user.ratings.items():
                dataset.append((str(user_id), str(movie_id), float(rating)))

        # w razie problemow odkomentowac sampling, wtedy dziala sprawnie
        # sample_size = int(0.5 * len(dataset))
        # dataset = random.sample(dataset, sample_size)

        df = pd.DataFrame(dataset, columns=['userId', 'movieId', 'rating'])
        self.train_data = Dataset.load_from_df(df, reader)

    def train_model(self):
        self.model = SVD(n_epochs=20)
        trainset = self.train_data.build_full_trainset()

        with tqdm(total=1, desc="Training SVD model") as pbar:
            self.model.fit(trainset)
            pbar.update(1)

    def rate(self, user_id, movie_id):
        try:
            prediction = self.model.predict(str(user_id), str(movie_id))
            return prediction.est
        except Exception as e:
            print(f"Prediction Error: {e}")
            return 2.5

    def __str__(self):
        return "SVD Recommender System by 151756"
    
class Sys151774(RatingSystem):
    def __init__(self):
        super().__init__()
        self.genre_avg_ratings: np.ndarray|None = None
        self.genres: dict[str, int] = {}
        self.movie_genres: dict[int, list[int]] = {}
        self._fit()

    def _fit(self):
        self._get_genres()
        genre_rating_totals = np.zeros(len(self.genres))
        genre_rating_counts = genre_rating_totals.copy()
        for movie, ratings in self.movie_ratings.items(): #dict[int, list[float]]
            i = self.movie_genres[movie]
            genre_rating_totals[i] += sum(ratings)
            genre_rating_counts[i] += len(ratings)
        self.genre_avg_ratings = genre_rating_totals / genre_rating_counts

    def _get_genres(self):
        counter: int = 0
        with open('../data/movie.csv', encoding='utf-8') as f:
            reader = csv.reader(f)
            reader.__next__()
            for line in reader:
                genres = line[2].split(sep='|')
                for genre in genres:
                    if not genre in self.genres:
                        self.genres[genre] = counter
                        counter += 1
                self.movie_genres[int(line[0])] = [self.genres[g_key] for g_key in genres]
    
    def rate(self, user, movie):
        user_genre_rating_totals = np.zeros(len(self.genres))
        user_genre_rating_counts = user_genre_rating_totals.copy()
        for key, val in user.ratings.items(): #dict[int, float]
            genres = self.movie_genres[key]
            user_genre_rating_totals[genres] += val
            user_genre_rating_counts[genres] += 1
        missing_indices = np.where(user_genre_rating_counts == 0)
        user_genre_rating_counts[missing_indices] = 1
        user_genre_rating_totals[missing_indices] = np.sum(user_genre_rating_totals) / np.sum(user_genre_rating_counts)
        user_genre_avg_ratings = user_genre_rating_totals / user_genre_rating_counts

        i = self.movie_genres[movie]
        movie_avg_rating = np.mean(self.movie_ratings[movie]) if len(self.movie_ratings[movie]) > 0 else 2.5
        movie_rating_stdev = np.std([x/5 for x in self.movie_ratings[movie]])
        #user_avg_rating = np.mean(list(user.ratings.values())) if len(user.ratings) > 0 else movie_avg_rating
        user_genre_avg_rating = np.sum(user_genre_avg_ratings[i])/len(i) if len(user.ratings) > 0 else movie_avg_rating
        if len(self.movie_ratings[movie]) == 0 and len(user.ratings) > 0:
            movie_avg_rating = user_genre_avg_rating
        user_movie_diff = user_genre_avg_rating - movie_avg_rating
        return user_movie_diff * movie_rating_stdev + movie_avg_rating
    
    def __str__(self):
        return 'Genre Average Weighted Rating (151774)'
    
class Sys151778(RatingSystem):
    def __init__(self):
        super().__init__()
        self.global_average = self.calculate_global_average()
        self.user_average = self.calculate_user_average()
        self.movie_average = self.calculate_movie_average()

    def calculate_global_average(self):
        """Calculates gloval average rating for all the movies"""
        total, count = 0, 0
        for rating in self.movie_ratings.values():
            total += sum(rating)
            count += len(rating)
        return total / count if count else 3.0

    def calculate_user_average(self):
        """Calculates user average rating"""
        user_avg = {}
        for user in self.users.values():
            ratings = list(user.ratings.values())
            user_avg[user.id] = sum(
                ratings) / len(ratings) if ratings else self.global_average
        return user_avg

    def calculate_movie_average(self):
        """Calculates movie average rating"""
        movie_avg = {}
        for movie_id, ratings in self.movie_ratings.items():
            movie_avg[movie_id] = sum(ratings) / len(ratings)
        return movie_avg

    def rate(self, user, movie_id):
        # fallback
        if movie_id not in self.movie_average:
            return self.user_average.get(user.id, self.global_average)

        deviations = []
        for m_id, rating in user.ratings.items():
            if m_id in self.movie_average:
                deviation = rating - self.movie_average[m_id]
                deviations.append(deviation)

        if not deviations:
            return self.movie_average[movie_id]

        avg_deviation = sum(deviations) / len(deviations)
        return self.movie_average[movie_id] + avg_deviation

    def __str__(self):
        return 'System created by 151778'
    
class Sys151833(RatingSystem):
    def __init__(self):
        self.tag_dict = {}
        super().__init__()
        self.GlobalAverageMovieRating = 0
        self.TotalMovies = 0
        for movie in self.movie_ratings:
            for rating  in self.movie_ratings[movie]:
                self.GlobalAverageMovieRating += rating
                self.TotalMovies += 1
        self.GlobalAverageMovieRating /= max(self.TotalMovies, 1)
        with open('../data/movie.csv', encoding='utf-8') as f:
            reader = csv.reader(f)
            reader.__next__()
            for l in reader:
                self.tag_dict[int(l[0])] = l[2].split("|")

    def rate(self, user, movie):
        if movie not in self.movie_ratings:
            return 2.5
        avg_movie = sum(self.movie_ratings[movie])/max(len(self.movie_ratings[movie]), 1)
        avg_user = self.calculate_avg_user(user.ratings, movie)
        if avg_user == 0 and avg_movie == 0:
            return self.GlobalAverageMovieRating
        if avg_user == 0:
            return (avg_movie + self.GlobalAverageMovieRating) / 2
        if avg_movie == 0:
            return (avg_user + self.GlobalAverageMovieRating) / 2
        return (avg_movie + avg_user) / 2

    def calculate_avg_user(self, user_ratings, movie: np.int64) -> int:
        movie_tags = self.tag_dict[movie.item()]
        movie_tags_set = set(movie_tags)
        rate_sum = 0
        rate_weight = 0
        for user_rating_movie in user_ratings:
            temp_movie_tags = self.tag_dict[user_rating_movie]
            weight = len(movie_tags_set.intersection(temp_movie_tags))
            rate_sum += (user_ratings[user_rating_movie] * weight)
            rate_weight += weight
        if rate_weight == 0:
            return 0
        else:
            return rate_sum/rate_weight

    def __str__(self):
        return 'MySystem (151833)'
    
class Sys151835(RatingSystem):
    def __init__(self):
        super().__init__()
        self.sum_of_avgs=0
        self.number_of_users=len(self.users)
        for user in self.users.values():
            if len(user.ratings)>0:
                user_avg=sum(user.ratings.values())/len(user.ratings)
                self.sum_of_avgs+=user_avg

        if self.number_of_users==0 or self.sum_of_avgs==0:
            self.global_avg_user_review=2.5
        else:
            self.global_avg_user_review=self.sum_of_avgs/self.number_of_users
        
                
    def rate(self, user, movie):
        n = len(user.ratings.values())
        m = len(self.movie_ratings[movie])
        if n == 0 and m==0:
            return self.global_avg_user_review
        else:
            self.avg_rating_for_this_user=sum(user.ratings.values())/n
            self.avg_rating_for_this_movie=sum(self.movie_ratings[movie])/m

            diff = self.avg_rating_for_this_user-self.global_avg_user_review
            return self.avg_rating_for_this_movie+diff
            
            
    def __str__(self):
        return 'System created by 151835'
        # lets calculate avg user rating and compare it to global avg user rating
        # because the person might be sceptical and pseeimist or WYBREDNA jak Naru sxjcflasdf
        
class Sys151850(RatingSystem):
    def __init__(self):
        super().__init__()
        
        self.movie_genres = {}        
        with open('../data/movie.csv', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            csv_reader.__next__()
            for line in csv_reader:
                self.movie_genres[int(line[0])] = line[2].split("|")
        
        self.GlobalAverageMovieRating = 0
        self.TotalMovies = 0
        for movie in self.movie_ratings:
            for rating  in self.movie_ratings[movie]:
                self.GlobalAverageMovieRating += rating
                self.TotalMovies += 1
        self.GlobalAverageMovieRating /= self.TotalMovies
        
    def rate(self, user, movie):
        
        
        genre_ratings=[]
        for movie_id, rating in user.ratings.items():
            if set(self.movie_genres[movie_id]) & set(self.movie_genres[movie]):
                genre_ratings.append(rating)
                
        if len(genre_ratings) < 5:
            avg_user_rating = np.mean(list(user.ratings.values()))
            user_bias = np.mean([rating - self.GlobalAverageMovieRating for rating in list(user.ratings.values())])
        else:
            avg_user_rating=np.mean(genre_ratings)
            user_bias = np.mean([rating - self.GlobalAverageMovieRating for rating in genre_ratings])
            
        movie_bias=np.mean([rating - self.GlobalAverageMovieRating for rating in self.movie_ratings[movie]])  
        prediction = self.GlobalAverageMovieRating+movie_bias+user_bias
        
        if not 0.5 <= prediction <= 5:
            return avg_user_rating            
        else:
            return prediction
    
    def __str__(self):
        return 'System created by 151850'
    
    
class Sys151851(RatingSystem):
    def __init__(self):
        super().__init__()
        self.global_avg = self._compute_global_avg()
        self.min_common = 3 

    def _compute_global_avg(self):
        total = 0
        count = 0
        for ratings in self.movie_ratings.values():
            total += sum(ratings)
            count += len(ratings)
        return total / count if count > 0 else 2.5

    def average_rating(self, user):
        if not user.ratings:
            return self.global_avg
        return sum(user.ratings.values()) / len(user.ratings)

    def pearson_similarity(self, user1, user2):
        common_movies = set(user1.ratings).intersection(user2.ratings)
        if len(common_movies) < self.min_common:
            return 0

        ratings1 = [user1.ratings[m] for m in common_movies]
        ratings2 = [user2.ratings[m] for m in common_movies]

        avg1 = sum(ratings1) / len(ratings1)
        avg2 = sum(ratings2) / len(ratings2)

        numerator = sum((r1 - avg1) * (r2 - avg2) for r1, r2 in zip(ratings1, ratings2))
        denominator1 = math.sqrt(sum((r1 - avg1) ** 2 for r1 in ratings1))
        denominator2 = math.sqrt(sum((r2 - avg2) ** 2 for r2 in ratings2))

        if denominator1 == 0 or denominator2 == 0:
            return 0

        return numerator / (denominator1 * denominator2)

    def rate(self, user, movie):
        if movie in user.ratings:
            return user.ratings[movie]

        if movie not in self.movie_ratings or not self.movie_ratings[movie]:
            return self.global_avg

        user_avg = self.average_rating(user)

        numerator = 0
        denominator = 0

        for other_user_id, other_user in self.users.items():
            if other_user_id == user.id or movie not in other_user.ratings:
                continue

            sim = self.pearson_similarity(user, other_user)
            if sim <= 0:
                continue

            other_avg = self.average_rating(other_user)
            adjusted = other_user.ratings[movie] - other_avg

            numerator += sim * adjusted
            denominator += abs(sim)

        if denominator == 0:
            return sum(self.movie_ratings[movie]) / len(self.movie_ratings[movie])

        predicted = user_avg + (numerator / denominator)
        return max(1.0, min(5.0, predicted))

    def __str__(self):
        return 'MyPearsonAdjusted (151851)'
    
class Sys151861(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        return (np.mean(np.array(self.movie_ratings[movie])) + np.mean(np.array(list(user.ratings.values())))) / 2
        
    def __str__(self):
        return 'Bestia Tytan Ostateczny (151861)'
    
    
class Sys151867(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        user_ratings = list(user.ratings.values())
        movie_ratings = self.movie_ratings.get(movie, [])
        if not user_ratings and not movie_ratings:
            return 2.5

        movie_mean = np.mean(movie_ratings) if movie_ratings else 2.5
        user_median = np.median(user_ratings) if user_ratings else 2.5

        return np.mean([movie_mean, user_median])

    def __str__(self):
        return 'System created by 151867'
    
class Sys151868(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movie_genres = {}

        # fill out a dictionary with move and it's genres
        with open('../data/movie.csv', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # header skip
            for line in csv_reader:
                movie_id = int(line[0])
                genres = line[2].split('|')
                self.movie_genres[movie_id] = genres

    def rate(self, user, movie_id):

        # should never happen
        if movie_id not in self.movie_genres:
            return 2.5

        movie_genres_list = self.movie_genres[movie_id]

        ratings_for_genre = []

        for rated_movie_id, rating in user.ratings.items():
            if rated_movie_id not in self.movie_genres:
                continue

            rated_movie_genres = self.movie_genres[rated_movie_id]
            if any(genre in movie_genres_list for genre in rated_movie_genres):
                ratings_for_genre.append(rating)

        if ratings_for_genre:
            return sum(ratings_for_genre) / len(ratings_for_genre)
        else:
            return 2.5

    def __str__(self):
        return 'System created by 151868'
    
    
class Sys151885(RatingSystem):
    def __init__(self):
        super().__init__()

        self.global_mean = 0.0
        self.user_biases = defaultdict(float)
        self.item_biases = defaultdict(float)

        self._calculate_baselines()

    def _calculate_baselines(self):
        all_ratings_sum = 0.0
        all_ratings_count = 0
        for movie_id, ratings in self.movie_ratings.items():
            if ratings:
                all_ratings_sum += sum(ratings)
                all_ratings_count += len(ratings)
        
        self.global_mean = all_ratings_sum / all_ratings_count if all_ratings_count > 0 else 2.5

        user_averages = defaultdict(float)
        for user_id, user_obj in self.users.items():
            ratings = list(user_obj.ratings.values())
            if ratings:
                user_averages[user_id] = sum(ratings) / len(ratings)

        for user_id, avg_rating in user_averages.items():
            self.user_biases[user_id] = avg_rating - self.global_mean


        for movie_id, ratings in self.movie_ratings.items():
             if ratings:
                 avg_item_rating = sum(ratings) / len(ratings)
                 self.item_biases[movie_id] = avg_item_rating - self.global_mean

    def rate(self, user, movie_id):
        user_id = user.id 

        user_b = self.user_biases.get(user_id, 0.0)
        item_b = self.item_biases.get(movie_id, 0.0)

        prediction = self.global_mean + user_b + item_b

        return np.clip(prediction, 0.5, 5.0) 

    def __str__(self):
        return 'System created by 151885'
    
    
class Sys151895(RatingSystem):
    def __init__(self):
        super().__init__()
        
        # Indeks użytkowników filmu: movie_id -> set(user_ids)
        self.movie_users = {movie_id: set() for movie_id in self.movie_ratings}
        for user_id, user in self.users.items():
            for movie_id in user.ratings:
                self.movie_users[movie_id].add(user_id)

        # Indeks gatunków filmu: movie_id -> set(genres)
        self.movie_genres = {}
        for movie_id in self.movie_ratings:
            movie = Movie.index.get(movie_id)
            if movie and movie.genres:
                self.movie_genres[movie_id] = set(movie.genres)
            else:
                self.movie_genres[movie_id] = set()

    def rate(self, user, movie_id):
        # Jeżeli użytkownik nie ma ocenionych filmów, zwróć średnią filmu lub domyślną wartość
        if not user.ratings:
            return np.mean(self.movie_ratings.get(movie_id, [2.5]))

        similarities = []
        ratings = []

        target_users = self.movie_users.get(movie_id, set())
        target_genres = self.movie_genres.get(movie_id, set())

        for other_movie_id, rating in user.ratings.items():
            if other_movie_id == movie_id:
                continue

            other_users = self.movie_users.get(other_movie_id, set())
            other_genres = self.movie_genres.get(other_movie_id, set())

            # Jaccard similarity (na podstawie użytkowników)
            user_intersection = len(target_users & other_users)
            user_union = len(target_users | other_users)
            jaccard_sim = user_intersection / user_union if user_union > 0 else 0

            # Overlap gatunków
            genre_intersection = len(target_genres & other_genres)
            genre_union = len(target_genres | other_genres)
            genre_sim = genre_intersection / genre_union if genre_union > 0 else 0

            # Całkowite podobieństwo
            similarity = 0.7 * jaccard_sim + 0.3 * genre_sim

            if similarity > 0:
                similarities.append(similarity)
                ratings.append(rating)

        if not similarities:
            return np.mean(self.movie_ratings.get(movie_id, [2.5]))

        return np.dot(similarities, ratings) / sum(similarities)

    def __str__(self):
        return 'Similarity + Genre Boosted System 151895'
    
    
class Sys152040(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movie_averages = {}
        for movie_id in self.movie_ratings:
            ratings = self.movie_ratings[movie_id]
            self.movie_averages[movie_id] = np.mean(ratings) if ratings else 2.5

    def rate(self, user, movie):
        if movie in user.ratings:
            return user.ratings[movie]
        
        user_ratings = list(user.ratings.values())
        user_average = np.mean(user_ratings) if user_ratings else 2.5
        movie_average = self.movie_averages.get(movie, 2.5)
        return 0.6 * movie_average + 0.4 * user_average

    def __str__(self):
        return 'System created by 152040'
    
    
class Sys152043(RatingSystem):
    def __init__(self):
        super().__init__()
        self.global_avg = self._compute_global_average()
        self.movie_bias = self._compute_movie_biases()
        self.user_bias = {}  

    def _compute_global_average(self):
        total = 0
        count = 0
        for ratings in self.movie_ratings.values():
            total += sum(ratings)
            count += len(ratings)
        return total / count if count > 0 else 2.5

    def _compute_movie_biases(self):
        biases = {}
        for movie, ratings in self.movie_ratings.items():
            if ratings:
                avg = sum(ratings) / len(ratings)
                biases[movie] = avg - self.global_avg
        return biases

    def _compute_user_bias(self, user):
        if user in self.user_bias:
            return self.user_bias[user]

        ratings = user.ratings
        if not ratings:
            self.user_bias[user] = 0
            return 0

        total_bias = 0
        count = 0
        for movie, rating in ratings.items():
            movie_b = self.movie_bias.get(movie, 0)
            total_bias += (rating - self.global_avg - movie_b)
            count += 1

        bias = total_bias / count if count > 0 else 0
        self.user_bias[user] = bias
        return bias

    def rate(self, user, movie):
        movie_b = self.movie_bias.get(movie, 0)
        user_b = self._compute_user_bias(user)
        rating = self.global_avg + movie_b + user_b
        return min(5.0, max(0.5, rating)) 

    def __str__(self):
        return 'MySystem (152043)'
    
    
class Sys152809(RatingSystem):
    def __init__(self):
        super().__init__()

    def _get_user_mean(self, user):
        ratings = list(user.ratings.values())
        if ratings:
            return sum(ratings) / len(ratings)
        return self._default_score()

    def _get_movie_mean(self, movie):
        scores = self.movie_ratings.get(movie, [])
        if scores:
            return float(np.mean(scores))
        return self._default_score()

    def _default_score(self):
        return 2.5

    def rate(self, user, movie):
        user_score = self._get_user_mean(user)
        movie_score = self._get_movie_mean(movie)
        return 0.5 * (user_score + movie_score)

    def __str__(self):
        return 'System created by 152809'
    
    
class Sys153029(RatingSystem):
    def __init__(self, weight=0.7):
        super().__init__()
        self.weight = weight


    def rate(self, user, movie):
        movie_ratings = self.movie_ratings.get(movie, [])
        if movie_ratings:
            avg_movie = sum(movie_ratings) / len(movie_ratings)
        else:
            avg_movie = 2.5

        user_ratings = list(user.ratings.values())
        if user_ratings:
            avg_user = sum(user_ratings) / len(user_ratings)
        else:
            avg_user = 2.5

        return self.weight * avg_movie + (1 - self.weight) * avg_user


    def __str__(self):
        return 'System created by 153029'