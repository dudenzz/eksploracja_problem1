from RatingSystem import RatingSystem
import csv


class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()

        # Parametry regularizacji / wag
        self.LAMBDA_MOVIE = 20.0
        self.LAMBDA_USER = 10.0
        self.LAMBDA_GENRE = 5.0

        self.W_MOVIE = 0.55
        self.W_USER = 0.30
        self.W_GENRE = 0.15

        # Wczytanie gatunków filmów
        self.movie_genres = {}
        with open('../data/movie.csv', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for line in csv_reader:
                movie_id = int(line[0])
                genres_raw = line[2].strip()
                if genres_raw == '(no genres listed)' or genres_raw == '':
                    self.movie_genres[movie_id] = set()
                else:
                    self.movie_genres[movie_id] = set(genres_raw.split('|'))

        # Średnia globalna
        total_sum = 0.0
        total_count = 0
        for movie_id in self.movie_ratings:
            ratings = self.movie_ratings[movie_id]
            total_sum += sum(ratings)
            total_count += len(ratings)

        self.global_mean = total_sum / total_count if total_count > 0 else 2.5

        # Regularizowane średnie filmów
        # movie_mean_reg[movie_id] = (sum_ratings + lambda * global_mean) / (n + lambda)
        self.movie_mean_reg = {}
        self.movie_count = {}
        for movie_id in self.movie_ratings:
            ratings = self.movie_ratings[movie_id]
            n = len(ratings)
            s = sum(ratings)
            self.movie_count[movie_id] = n
            self.movie_mean_reg[movie_id] = (s + self.LAMBDA_MOVIE * self.global_mean) / (n + self.LAMBDA_MOVIE)

        # Statystyki gatunków w całym zbiorze treningowym
        self.genre_sum = {}
        self.genre_count = {}
        for movie_id in self.movie_ratings:
            ratings = self.movie_ratings[movie_id]
            genres = self.movie_genres.get(movie_id, set())
            if not genres:
                continue

            for r in ratings:
                for g in genres:
                    if g not in self.genre_sum:
                        self.genre_sum[g] = 0.0
                        self.genre_count[g] = 0
                    self.genre_sum[g] += r
                    self.genre_count[g] += 1

        self.genre_mean = {}
        for g in self.genre_sum:
            self.genre_mean[g] = self.genre_sum[g] / self.genre_count[g]

    def _clip_rating(self, value):
        # W MovieLens często występują oceny od 0.5 do 5.0.
        # przy wymaganiu 1-5, można zmienić dolny limit na 1.0.
        if value < 0.5:
            return 0.5
        if value > 5.0:
            return 5.0
        return value

    def _regularized_user_mean(self, user):
        n = len(user.ratings)
        if n == 0:
            return self.global_mean

        s = sum(user.ratings.values())
        return (s + self.LAMBDA_USER * self.global_mean) / (n + self.LAMBDA_USER)

    def _movie_prior(self, movie):
        if movie in self.movie_mean_reg:
            return self.movie_mean_reg[movie]

        # Jeśli film nie ma ocen treningowych, próbujemy użyć średniej gatunkowej.
        genres = self.movie_genres.get(movie, set())
        if genres:
            vals = [self.genre_mean[g] for g in genres if g in self.genre_mean]
            if len(vals) > 0:
                return sum(vals) / len(vals)

        return self.global_mean

    def _user_genre_profile(self, user, movie):
        """
        Szacuje, jak bardzo użytkownik lubi gatunki filmu docelowego.
        Liczymy ważoną średnią ocen użytkownika dla filmów o pokrywających się gatunkach.
        """
        target_genres = self.movie_genres.get(movie, set())
        if len(target_genres) == 0 or len(user.ratings) == 0:
            return None

        weighted_sum = 0.0
        weight_total = 0.0

        for seen_movie, rating in user.ratings.items():
            seen_genres = self.movie_genres.get(seen_movie, set())
            if not seen_genres:
                continue

            overlap = len(target_genres.intersection(seen_genres))
            if overlap == 0:
                continue

            # lekka waga podobieństwa gatunkowego
            union = len(target_genres.union(seen_genres))
            sim = overlap / union if union > 0 else 0.0

            weighted_sum += sim * rating
            weight_total += sim

        if weight_total == 0.0:
            return None

        # regularizacja względem średniej użytkownika
        user_mean = self._regularized_user_mean(user)
        return (weighted_sum + self.LAMBDA_GENRE * user_mean) / (weight_total + self.LAMBDA_GENRE)

    def rate(self, user, movie):
        """
        Zwraca przewidywaną ocenę dla użytkownika 'user' i filmu 'movie'.
        """
        # Gdyby film nadal był w user_copy (zabezpieczenie), zwracamy prawdziwą ocenę
        if movie in user.ratings:
            return user.ratings[movie]

        movie_prior = self._movie_prior(movie)
        user_prior = self._regularized_user_mean(user)
        genre_profile = self._user_genre_profile(user, movie)

        if genre_profile is None:
            pred = self.W_MOVIE * movie_prior + self.W_USER * user_prior + (1.0 - self.W_MOVIE - self.W_USER) * self.global_mean
        else:
            pred = self.W_MOVIE * movie_prior + self.W_USER * user_prior + self.W_GENRE * genre_profile

        return self._clip_rating(pred)

    def __str__(self):
      return 'System created by 156027 and 155829'
        # return 'Hybrid Genre-Mean (156027 i 155829)'