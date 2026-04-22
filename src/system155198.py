"""
Bazowy moduł oceny dla filmów i użytkowników
"""
import random
from RatingSystem import RatingSystem
from RatingLib import User

class BiasRatingSystem(RatingSystem):
    """
    Klasa reprezentująca działanie Bias Rating System z buforowaniem
    i sekwencyjnym obliczaniem biasów

    Ze względu na parametry faktoryzacji oraz wprowadzony trening, zadanie może rozwiązywać się ponad 20 min.
    """

    def __init__(self):
        super().__init__()
        # parametr regularyzacji występuje w mianowniku
        # w celu obniżenia biasu przy małej liczbie oceny użytkownika
        self.lambda_m = 35
        self.lambda_u = 18

        # Parametry faktoryzacji
        self.k = 25
        self.lr = 0.008
        self.lamb_b = 0.015
        self.lamb_f = 0.02
        self.epochs = 5
        self.lr_decay = 0.85

        # Bufory i struktury
        self.mean_global = None
        self.movie_biases = {}
        self.user_biases = {}
        self.user_factors = {}
        self.movie_factors = {}

        # Treining i init
        self.calculate_global_mean()
        self._train()

    def calculate_global_mean(self):
        """
        Ta metoda zwraca globalną średnią ocen wszystkich filmów.
        """
        if self.mean_global is not None:
            return self.mean_global

        total = 0
        count = 0

        for ratings in self.movie_ratings.values():
            total += sum(ratings)
            count += len(ratings)

        if count > 0:
            self.mean_global = total / count
        else:
            self.mean_global = 2.5
        return self.mean_global

    def calculate_user_bias(self, user):
        """
        Ta metoda zwraca bias użytkownika,
        jego wskaźnik podatności na ocenę + (pozytywnie) - (negatywnie).
        """
        if user not in self.user_biases:
            ratings_dictionary = user.ratings
            n = len(ratings_dictionary)

            if n == 0:
                self.user_biases[user] = 0
            else:
                g_mean = self.calculate_global_mean()
                diff_sum = sum(
                    r - g_mean - self.calculate_movie_bias(m)
                    for m, r in ratings_dictionary.items()
                )
                self.user_biases[user] = diff_sum / (self.lambda_u + n)
        return self.user_biases[user]

    def calculate_movie_bias(self, movie):
        """
        Ta metoda zwraca bias filmu czyli jego wskaźnik ocen + (pozytywnie) - (negatywnie).
        """
        if movie not in self.movie_biases:
            m_id = getattr(movie, "id", movie)
            ratings = self.movie_ratings.get(movie, [])
            n = len(ratings)
            if n == 0:
                self.movie_biases[movie] = 0
            else:
                g_mean = self.calculate_global_mean()
                diff_sum = sum(r - g_mean for r in ratings)
                self.movie_biases[m_id] = diff_sum / (self.lambda_m + n)
        return self.movie_biases.get(getattr(movie, "id", movie), 0)

    def _get_train_data(self):
        """
        Przygotowanie danych do uczenia
        """
        try:
            from test_users import test_users
        except ImportError:
            test_users = []

        train_data = []
        scale = 0.1

        for user_id, user_obj in User.index.items():
            u_ratings = list(user_obj.ratings.values())
            self.user_biases[user_id] = (
                (sum(u_ratings) / len(u_ratings) - self.mean_global) if u_ratings else 0
            )
            self.user_factors[user_id] = [
                random.uniform(-scale, scale) for _ in range(self.k)
            ]

            for movie_ref, rating in user_obj.ratings.items():
                m_id = getattr(movie_ref, "id", movie_ref)
                train_data.append((user_id, m_id, rating))

                if m_id not in self.movie_factors:
                    self.movie_factors[m_id] = [
                        random.uniform(-scale, scale) for _ in range(self.k)
                    ]
                    if m_id not in self.movie_biases:
                        self.calculate_movie_bias(m_id)

        for t_u_id, t_m_id in test_users:
            if t_u_id not in self.user_factors:
                self.user_factors[t_u_id] = [
                    random.uniform(-scale, scale) for _ in range(self.k)
                ]
                self.user_biases[t_u_id] = 0
            if t_m_id not in self.movie_factors:
                self.movie_factors[t_m_id] = [
                    random.uniform(-scale, scale) for _ in range(self.k)
                ]
                self.calculate_movie_bias(t_m_id)

        return train_data

    def _train(self):
        """
        Główna pętla uczenia
        """
        data = self._get_train_data()
        lr = self.lr

        print(f"[System] Start treningu: {len(data)} próbek, {self.epochs} epok.")

        for epoch in range(self.epochs):
            random.shuffle(data)
            print(f"  -> Początek epoki {epoch + 1}")
            for u_id, m_id, r in data:
                p_u = self.user_factors[u_id]
                q_m = self.movie_factors[m_id]

                interaction = sum(p_u[i] * q_m[i] for i in range(self.k))
                prediction = (
                    self.mean_global
                    + self.user_biases[u_id]
                    + self.movie_biases[m_id]
                    + interaction
                )

                error = r - prediction

                self.user_biases[u_id] += lr * (
                    error - self.lamb_b * self.user_biases[u_id]
                )
                self.movie_biases[m_id] += lr * (
                    error - self.lamb_b * self.movie_biases[m_id]
                )

                for i in range(self.k):
                    p_old = p_u[i]
                    p_u[i] += lr * (error * q_m[i] - self.lamb_f * p_u[i])
                    q_m[i] += lr * (error * p_old - self.lamb_f * q_m[i])

            lr *= self.lr_decay
            print(f"  -> Epoka {epoch + 1} zakończona (lr={lr:.5f})")

    def clamp_rating(self, rating):
        """
        Ta funkcja normalizuje ocene systemu do 1-5.
        """
        return max(1, min(5, rating))

    def rate(self, user, movie):
        """
        Ta metoda zwraca rating w skali 1-5.
        Jest to ocena przyznana przez użytkownika 'user' filmowi 'movie'.
        """
        u_id = getattr(user, "id", user)
        m_id = getattr(movie, "id", movie)

        if hasattr(user, "ratings") and movie in user.ratings:
            return user.ratings[movie]

        b_u = self.user_biases.get(u_id, 0)
        b_m = self.calculate_movie_bias(m_id)

        p_u = self.user_factors.get(u_id)
        q_m = self.movie_factors.get(m_id)

        prediction = self.mean_global + b_u + b_m

        if p_u is not None and q_m is not None:
            interaction = sum(p_u[i] * q_m[i] for i in range(self.k))
            prediction += interaction

        return self.clamp_rating(prediction)

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return "System created by 155198 and 155921""""
Bazowy moduł oceny dla filmów i użytkowników
"""
import random
from RatingSystem import RatingSystem
from RatingLib import User

class BiasRatingSystem(RatingSystem):
    """
    Klasa reprezentująca działanie Bias Rating System z buforowaniem
    i sekwencyjnym obliczaniem biasów

    Ze względu na parametry faktoryzacji oraz wprowadzony trening, zadanie może rozwiązywać się ponad 20 min.
    """

    def __init__(self):
        super().__init__()
        # parametr regularyzacji występuje w mianowniku
        # w celu obniżenia biasu przy małej liczbie oceny użytkownika
        self.lambda_m = 35
        self.lambda_u = 18

        # Parametry faktoryzacji
        self.k = 25
        self.lr = 0.008
        self.lamb_b = 0.015
        self.lamb_f = 0.02
        self.epochs = 5
        self.lr_decay = 0.85

        # Bufory i struktury
        self.mean_global = None
        self.movie_biases = {}
        self.user_biases = {}
        self.user_factors = {}
        self.movie_factors = {}

        # Treining i init
        self.calculate_global_mean()
        self._train()

    def calculate_global_mean(self):
        """
        Ta metoda zwraca globalną średnią ocen wszystkich filmów.
        """
        if self.mean_global is not None:
            return self.mean_global

        total = 0
        count = 0

        for ratings in self.movie_ratings.values():
            total += sum(ratings)
            count += len(ratings)

        if count > 0:
            self.mean_global = total / count
        else:
            self.mean_global = 2.5
        return self.mean_global

    def calculate_user_bias(self, user):
        """
        Ta metoda zwraca bias użytkownika,
        jego wskaźnik podatności na ocenę + (pozytywnie) - (negatywnie).
        """
        if user not in self.user_biases:
            ratings_dictionary = user.ratings
            n = len(ratings_dictionary)

            if n == 0:
                self.user_biases[user] = 0
            else:
                g_mean = self.calculate_global_mean()
                diff_sum = sum(
                    r - g_mean - self.calculate_movie_bias(m)
                    for m, r in ratings_dictionary.items()
                )
                self.user_biases[user] = diff_sum / (self.lambda_u + n)
        return self.user_biases[user]

    def calculate_movie_bias(self, movie):
        """
        Ta metoda zwraca bias filmu czyli jego wskaźnik ocen + (pozytywnie) - (negatywnie).
        """
        if movie not in self.movie_biases:
            m_id = getattr(movie, "id", movie)
            ratings = self.movie_ratings.get(movie, [])
            n = len(ratings)
            if n == 0:
                self.movie_biases[movie] = 0
            else:
                g_mean = self.calculate_global_mean()
                diff_sum = sum(r - g_mean for r in ratings)
                self.movie_biases[m_id] = diff_sum / (self.lambda_m + n)
        return self.movie_biases.get(getattr(movie, "id", movie), 0)

    def _get_train_data(self):
        """
        Przygotowanie danych do uczenia
        """
        try:
            from test_users import test_users
        except ImportError:
            test_users = []

        train_data = []
        scale = 0.1

        for user_id, user_obj in User.index.items():
            u_ratings = list(user_obj.ratings.values())
            self.user_biases[user_id] = (
                (sum(u_ratings) / len(u_ratings) - self.mean_global) if u_ratings else 0
            )
            self.user_factors[user_id] = [
                random.uniform(-scale, scale) for _ in range(self.k)
            ]

            for movie_ref, rating in user_obj.ratings.items():
                m_id = getattr(movie_ref, "id", movie_ref)
                train_data.append((user_id, m_id, rating))

                if m_id not in self.movie_factors:
                    self.movie_factors[m_id] = [
                        random.uniform(-scale, scale) for _ in range(self.k)
                    ]
                    if m_id not in self.movie_biases:
                        self.calculate_movie_bias(m_id)

        for t_u_id, t_m_id in test_users:
            if t_u_id not in self.user_factors:
                self.user_factors[t_u_id] = [
                    random.uniform(-scale, scale) for _ in range(self.k)
                ]
                self.user_biases[t_u_id] = 0
            if t_m_id not in self.movie_factors:
                self.movie_factors[t_m_id] = [
                    random.uniform(-scale, scale) for _ in range(self.k)
                ]
                self.calculate_movie_bias(t_m_id)

        return train_data

    def _train(self):
        """
        Główna pętla uczenia
        """
        data = self._get_train_data()
        lr = self.lr

        print(f"[System] Start treningu: {len(data)} próbek, {self.epochs} epok.")

        for epoch in range(self.epochs):
            random.shuffle(data)
            print(f"  -> Początek epoki {epoch + 1}")
            for u_id, m_id, r in data:
                p_u = self.user_factors[u_id]
                q_m = self.movie_factors[m_id]

                interaction = sum(p_u[i] * q_m[i] for i in range(self.k))
                prediction = (
                    self.mean_global
                    + self.user_biases[u_id]
                    + self.movie_biases[m_id]
                    + interaction
                )

                error = r - prediction

                self.user_biases[u_id] += lr * (
                    error - self.lamb_b * self.user_biases[u_id]
                )
                self.movie_biases[m_id] += lr * (
                    error - self.lamb_b * self.movie_biases[m_id]
                )

                for i in range(self.k):
                    p_old = p_u[i]
                    p_u[i] += lr * (error * q_m[i] - self.lamb_f * p_u[i])
                    q_m[i] += lr * (error * p_old - self.lamb_f * q_m[i])

            lr *= self.lr_decay
            print(f"  -> Epoka {epoch + 1} zakończona (lr={lr:.5f})")

    def clamp_rating(self, rating):
        """
        Ta funkcja normalizuje ocene systemu do 1-5.
        """
        return max(1, min(5, rating))

    def rate(self, user, movie):
        """
        Ta metoda zwraca rating w skali 1-5.
        Jest to ocena przyznana przez użytkownika 'user' filmowi 'movie'.
        """
        u_id = getattr(user, "id", user)
        m_id = getattr(movie, "id", movie)

        if hasattr(user, "ratings") and movie in user.ratings:
            return user.ratings[movie]

        b_u = self.user_biases.get(u_id, 0)
        b_m = self.calculate_movie_bias(m_id)

        p_u = self.user_factors.get(u_id)
        q_m = self.movie_factors.get(m_id)

        prediction = self.mean_global + b_u + b_m

        if p_u is not None and q_m is not None:
            interaction = sum(p_u[i] * q_m[i] for i in range(self.k))
            prediction += interaction

        return self.clamp_rating(prediction)

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return "System created by 155198 and 155921"