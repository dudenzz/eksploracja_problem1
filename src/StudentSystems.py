from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie
import tqdm
import csv

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
    


class Sys151868(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movie_genres = {}

        # fill out a dictionary with move and it's genres
        with open('data/movie.csv', encoding='utf-8') as file:
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


class Sys151861(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        return (np.mean(np.array(self.movie_ratings[movie])) + np.mean(np.array(list(user.ratings.values())))) / 2
        
    def __str__(self):
        return 'Bestia Tytan Ostateczny (151861)'
    
    
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
        with open('data/movie.csv', encoding='utf-8') as f:
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
    
from RatingSystem import RatingSystem
import numpy as np

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