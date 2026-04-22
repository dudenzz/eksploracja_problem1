from RatingSystem import RatingSystem
import test_users
import csv
import math


class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movie_genres = {}
        self.movie_to_user_ratings = {}
        self.item_sim_cache = {}

        test_set = {tuple(pair) for pair in test_users.test_pairs}

        try:
            with open('data/movie.csv', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for line in csv_reader:
                    if not line: continue
                    m_id = int(line[0])
                    self.movie_genres[m_id] = line[2].split('|') if len(line) > 2 else []
        except FileNotFoundError:
            pass

        for u_id, user_obj in self.users.items():
            for m_id, rating in user_obj.ratings.items():
                if (u_id, m_id) in test_set:
                    continue
                if m_id not in self.movie_to_user_ratings:
                    self.movie_to_user_ratings[m_id] = {}
                self.movie_to_user_ratings[m_id][u_id] = rating

    def _get_user_mean(self, ratings_dict):
        if not ratings_dict: return 2.5
        return sum(ratings_dict.values()) / len(ratings_dict)

    def _item_sim(self, m_a, m_b):
        pair = tuple(sorted((m_a, m_b)))
        if pair in self.item_sim_cache: return self.item_sim_cache[pair]

        r_a = self.movie_to_user_ratings.get(m_a, {})
        r_b = self.movie_to_user_ratings.get(m_b, {})
        common = set(r_a.keys()) & set(r_b.keys())

        if len(common) < 2: return 0.0

        dot, n_a, n_b = 0.0, 0.0, 0.0
        for u_id in common:
            dot += r_a[u_id] * r_b[u_id]
            n_a += r_a[u_id] ** 2
            n_b += r_b[u_id] ** 2

        sim = dot / (math.sqrt(n_a) * math.sqrt(n_b)) if n_a * n_b > 0 else 0.0
        self.item_sim_cache[pair] = sim
        return sim

    def _user_sim(self, r_a, r_b):
        common = set(r_a.keys()) & set(r_b.keys())
        if not common: return 0.0
        m_a, m_b = self._get_user_mean(r_a), self._get_user_mean(r_b)
        dot, n_a, n_b = 0.0, 0.0, 0.0
        for m_id in common:
            da, db = r_a[m_id] - m_a, r_b[m_id] - m_b
            dot += da * db
            n_a += da ** 2
            n_b += db ** 2
        return dot / (math.sqrt(n_a) * math.sqrt(n_b)) if n_a * n_b > 0 else 0.0

    def rate(self, user, movie):
        test_set = {tuple(pair) for pair in test_users.test_pairs}
        clean_user_ratings = {m: r for m, r in user.ratings.items() if (user.id, m) not in test_set}

        ib_num, ib_den, count_ib = 0.0, 0.0, 0
        for r_m_id, rating in clean_user_ratings.items():
            sim = self._item_sim(movie, r_m_id)
            if sim > 0:
                ib_num += sim * rating
                ib_den += sim
                count_ib += 1
        ib_res = (ib_num / ib_den, min(count_ib / 10, 1.0)) if ib_den > 0 else (None, 0)

        potential_users = self.movie_to_user_ratings.get(movie, {})
        ub_num, ub_den, count_ub = 0.0, 0.0, 0
        u_mean = self._get_user_mean(clean_user_ratings)
        for o_u_id, o_rating in potential_users.items():
            other_user = self.users[o_u_id]
            sim = self._user_sim(clean_user_ratings, other_user.ratings)
            if sim > 0:
                ub_num += sim * (o_rating - self._get_user_mean(other_user.ratings))
                ub_den += sim
                count_ub += 1
        ub_res = (u_mean + (ub_num / ub_den), min(count_ub / 15, 1.0)) if ub_den > 0 else (None, 0)

        t_gen = set(self.movie_genres.get(movie, []))
        gn_num, gn_den = 0.0, 0.0
        for r_m_id, rating in clean_user_ratings.items():
            weight = len(t_gen.intersection(set(self.movie_genres.get(r_m_id, []))))
            gn_num += rating * weight
            gn_den += weight
        gn_res = (gn_num / gn_den, min(gn_den / 15, 0.5)) if gn_den > 0 else (None, 0)

        results = []
        if ib_res[0] is not None: results.append((ib_res[0], ib_res[1] * 3.0))
        if ub_res[0] is not None: results.append((ub_res[0], ub_res[1] * 2.0))
        if gn_res[0] is not None: results.append((gn_res[0], gn_res[1] * 1.0))

        if results:
            final_val = sum(v * w for v, w in results) / sum(w for v, w in results)
            return max(0.5, min(5.0, final_val))

        u_avg = u_mean
        m_ratings = self.movie_ratings.get(movie, [])
        m_avg = sum(m_ratings) / len(m_ratings) if m_ratings else 2.5
        return (u_avg + m_avg) / 2.0

    def __str__(self):
        return 'System created by 155974, 155874 and 155879'