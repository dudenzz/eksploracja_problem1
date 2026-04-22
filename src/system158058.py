from collections import defaultdict
from math import sqrt
from RatingSystem import RatingSystem


class MySystem(RatingSystem):
    USER_REG = 15.0
    MOVIE_REG = 20.0

    SIM_MIN_COMMON = 4
    SIM_SHRINK = 10.0
    MAX_USER_MOVIES = 30
    CF_WEIGHT = 0.35

    def __init__(self):
        super().__init__()

        self.movie_sum = defaultdict(float)
        self.movie_cnt = defaultdict(int)

        total_sum = 0.0
        total_cnt = 0

        for movie_id, ratings in self.movie_ratings.items():
            if not ratings:
                continue
            movie_id = int(movie_id)
            s = float(sum(ratings))
            c = len(ratings)
            self.movie_sum[movie_id] = s
            self.movie_cnt[movie_id] = c
            total_sum += s
            total_cnt += c

        self.global_mean = total_sum / total_cnt if total_cnt else 2.5

        self.movie_bias = {}
        for movie_id, count in self.movie_cnt.items():
            s = self.movie_sum[movie_id]
            self.movie_bias[movie_id] = (s - count * self.global_mean) / (self.MOVIE_REG + count)

        self.movie_users = defaultdict(dict)
        for user_id, user in self.users.items():
            for movie_id, rating in user.ratings.items():
                self.movie_users[int(movie_id)][user_id] = float(rating)

    def build_user_bias(self, user):
        if not user.ratings:
            return 0.0
        bias_sum = 0.0
        bias_cnt = 0
        for movie_id, rating in user.ratings.items():
            movie_id = int(movie_id)
            bias_sum += float(rating) - self.global_mean - self.movie_bias.get(movie_id, 0.0)
            bias_cnt += 1
        return bias_sum / (self.USER_REG + bias_cnt)

    @staticmethod
    def clip(x):
        if x < 0.5:
            return 0.5
        if x > 5.0:
            return 5.0
        return x

    @staticmethod
    def normalize_to_half_step(x):
        x = MySystem.clip(float(x))
        return int(x * 2.0 + 0.5) / 2.0

    def baseline_for_movie(self, user_bias, movie_id):
        return self.global_mean + self.movie_bias.get(movie_id, 0.0) + user_bias

    def item_similarity(self, movie_a, movie_b):
        users_a = self.movie_users.get(movie_a)
        users_b = self.movie_users.get(movie_b)

        if not users_a or not users_b:
            return 0.0
        
        if len(users_a) < len(users_b):
            smaller, bigger = users_a, users_b
        else:
            smaller, bigger = users_b, users_a

        common_cnt = 0
        num = 0.0
        den_a = 0.0
        den_b = 0.0

        base_a = self.global_mean + self.movie_bias.get(movie_a, 0.0)
        base_b = self.global_mean + self.movie_bias.get(movie_b, 0.0)

        for user_id, ra_raw in smaller.items():
            rb_raw = bigger.get(user_id)
            if rb_raw is None:
                continue

            ra = users_a[user_id] - base_a
            rb = users_b[user_id] - base_b

            num += ra * rb
            den_a += ra * ra
            den_b += rb * rb
            common_cnt += 1

        if common_cnt < self.SIM_MIN_COMMON:
            return 0.0
        if den_a <= 1e-12 or den_b <= 1e-12:
            return 0.0

        sim = num / (sqrt(den_a) * sqrt(den_b))
        sim *= common_cnt / (common_cnt + self.SIM_SHRINK)
        return sim

    def item_correction(self, user, target_movie, user_bias):
        num = 0.0
        den = 0.0
        used = 0

        for other_movie, rating in user.ratings.items():
            other_movie = int(other_movie)
            if other_movie == target_movie:
                continue

            sim = self.item_similarity(target_movie, other_movie)
            if sim <= 0.0:
                continue

            base_other = self.baseline_for_movie(user_bias, other_movie)
            residual = float(rating) - base_other

            num += sim * residual
            den += abs(sim)

            used += 1
            if used >= self.MAX_USER_MOVIES:
                break

        if den <= 1e-12:
            return 0.0

        return num / den

    def rate(self, user, movie):
        movie_id = int(movie)

        movie_bias = self.movie_bias.get(movie_id, 0.0)
        user_bias = self.build_user_bias(user)

        base = self.global_mean + movie_bias + user_bias

        correction = self.item_correction(user, movie_id, user_bias)

        rate = base + self.CF_WEIGHT * correction
        return self.normalize_to_half_step(rate)

    def __str__(self):
        return "System created by 158058 and 155077"