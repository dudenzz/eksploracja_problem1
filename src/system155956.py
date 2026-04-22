import numpy as np
import heapq
from collections import defaultdict
from RatingSystem import RatingSystem
import test_users


class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        # Pary testowe, których nie wolno użyć do budowy modelu
        self.test_pairs_set = {(int(u), int(m)) for u, m in test_users.test_pairs}
        self.test_movies_by_user = defaultdict(set)
        for u, m in self.test_pairs_set:
            self.test_movies_by_user[u].add(m)

        # Oceny użytkowników po usunięciu par testowych
        self.user_clean_ratings = {}
        for u_id, user_obj in self.users.items():
            blocked = self.test_movies_by_user.get(int(u_id), set())
            if not blocked:
                self.user_clean_ratings[u_id] = dict(user_obj.ratings)
            else:
                self.user_clean_ratings[u_id] = {
                    m_id: r for m_id, r in user_obj.ratings.items() if int(m_id) not in blocked
                }

        self.k_neighbors = 35
        self.min_common_movies = 2
        self.sim_shrink_reg = 6.0
        self.blend_reg = 5.0
        self.user_fallback_weight = 0.30

        # Średnie ocen filmów i średnia globalna
        ratings_sum = 0.0
        ratings_count = 0
        self.movie_means = {}
        for movie_id, ratings in self.movie_ratings.items():
            if ratings:
                movie_sum = float(sum(ratings))
                movie_count = len(ratings)
                self.movie_means[movie_id] = movie_sum / movie_count
                ratings_sum += movie_sum
                ratings_count += movie_count
        self.global_avg = (ratings_sum / ratings_count) if ratings_count else 3.5

        # Indeks odwrócony: film -> lista (użytkownik, ocena)
        self.movie_to_user_ratings = defaultdict(list)
        for u_id, ratings in self.user_clean_ratings.items():
            for m_id, rat in ratings.items():
                self.movie_to_user_ratings[m_id].append((u_id, rat))

        # Średnia ocena każdego użytkownika na oczyszczonych danych
        self.user_means = {}
        for u_id, ratings in self.user_clean_ratings.items():
            u_ratings = list(ratings.values())
            if u_ratings:
                self.user_means[u_id] = float(sum(u_ratings) / len(u_ratings))
            else:
                self.user_means[u_id] = self.global_avg

    def _get_centered_cosine_similarity(self, ratings1, ratings2, mean1, mean2):
        # Liczymy po mniejszym słowniku, żeby było szybciej
        if len(ratings1) > len(ratings2):
            ratings1, ratings2 = ratings2, ratings1

        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        common_count = 0
        for movie_id, rating in ratings1.items():
            other = ratings2.get(movie_id)
            if other is not None:
                c1 = rating - mean1
                c2 = other - mean2
                dot_product += c1 * c2
                norm1 += c1 * c1
                norm2 += c2 * c2
                common_count += 1

        # Odrzucamy zbyt małą część wspólną albo zerową normę
        if common_count < self.min_common_movies or norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0, common_count

        # Similarity z wygaszaniem dla małej liczby wspólnych ocen
        sim = dot_product / ((norm1 ** 0.5) * (norm2 ** 0.5))
        shrink = common_count / (common_count + self.sim_shrink_reg)
        return sim * shrink, common_count

    def rate(self, user, movie_id):
        # movie może być obiektem Movie albo id
        m_id = movie_id.id if hasattr(movie_id, "id") else int(movie_id)

        potential_neighbors = self.movie_to_user_ratings.get(m_id, [])
        # Dodatkowe czyszczenie ocen aktywnego użytkownika z par testowych
        blocked = self.test_movies_by_user.get(int(user.id), set())
        if blocked:
            user_ratings_dict = {
                mid: rating for mid, rating in user.ratings.items() if int(mid) not in blocked
            }
        else:
            user_ratings_dict = user.ratings

        movie_mean = self.movie_means.get(m_id, self.global_avg)
        if user_ratings_dict:
            active_mean = float(sum(user_ratings_dict.values()) / len(user_ratings_dict))
            baseline = self.user_fallback_weight * active_mean + (1.0 - self.user_fallback_weight) * movie_mean
        else:
            active_mean = self.global_avg
            baseline = movie_mean

        # Fallback gdy brak danych do collaborative filtering
        if not potential_neighbors or not user_ratings_dict:
            prediction = round(baseline * 2) / 2
            return max(0.5, min(5.0, float(prediction)))

        similarities = []

        for neighbor_id, neighbor_rating in potential_neighbors:
            neighbor_ratings = self.user_clean_ratings.get(neighbor_id)
            if neighbor_ratings:
                neighbor_mean = self.user_means.get(neighbor_id, self.global_avg)
                sim, _common = self._get_centered_cosine_similarity(
                    user_ratings_dict, neighbor_ratings, active_mean, neighbor_mean
                )
                if sim > 0:
                    similarities.append((sim, neighbor_rating - neighbor_mean))

        # Wybór top-k sąsiadów
        top_k = heapq.nlargest(self.k_neighbors, similarities, key=lambda x: x[0])
        if not top_k:
            prediction = round(baseline * 2) / 2
            return max(0.5, min(5.0, float(prediction)))

        # Predykcja CF: średnia aktywnego usera + ważone odchylenia sąsiadów
        weighted_dev_sum = sum(sim * dev for sim, dev in top_k)
        sum_of_weights = sum(abs(sim) for sim, _dev in top_k)
        if sum_of_weights <= 0.0:
            pred_cf = active_mean
        else:
            pred_cf = active_mean + (weighted_dev_sum / sum_of_weights)

        # Mieszanie predykcji CF z baseline
        blend_weight = sum_of_weights / (sum_of_weights + self.blend_reg)
        prediction = blend_weight * pred_cf + (1.0 - blend_weight) * baseline

        # Krok co 0.5 i zakres 0.5-5.0
        prediction = round(prediction * 2) / 2
        return max(0.5, min(5.0, float(prediction)))

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return "System created by 155898 and 156021 and 155934"