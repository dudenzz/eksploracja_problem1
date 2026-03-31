from collections import defaultdict
from RatingSystem import RatingSystem


class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movie_sum = defaultdict(float)
        self.movie_cnt = defaultdict(int)

        total_sum = 0.0
        total_cnt = 0

        for movie_id, ratings in self.movie_ratings.items():
            if not ratings:
                continue
            s = float(sum(ratings))
            c = len(ratings)
            self.movie_sum[int(movie_id)] = s
            self.movie_cnt[int(movie_id)] = c
            total_sum += s
            total_cnt += c

        self.global_mean = total_sum / total_cnt if total_cnt else 2.5

    @staticmethod
    def _clip(x):
        if x < 0.5:
            return 0.5
        if x > 5.0:
            return 5.0
        return x

    @staticmethod
    def _normalize_to_half_step(x):
        # 0.5, 1.0, 1.5, ..., 5.0
        x = MySystem._clip(float(x))
        return int(x * 2.0 + 0.5) / 2.0

    def user_dif(self, user):
        diff_sum = 0.0
        diff_cnt = 0

        for movie_id, rating in user.ratings.items():
            movie_id = int(movie_id)
            c = self.movie_cnt.get(movie_id, 0)
            if c == 0:
                continue
            s = self.movie_sum[movie_id]
            others_mean = s / c
            diff_sum += float(rating) - others_mean
            diff_cnt += 1

        if diff_cnt == 0:
            return 0.0
        return diff_sum / diff_cnt

    def rate(self, user, movie):
        movie_id = int(movie)
        n = self.movie_cnt.get(movie_id, 0)
        if n == 0:
            return self.global_mean

        others_mean = self.movie_sum[movie_id] / n
        rate = others_mean + self.user_dif(user)
        return self._normalize_to_half_step(rate)

    def __str__(self):
        return "System created by 158058 and 155077"

