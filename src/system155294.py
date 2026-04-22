"""
# W klasie MySystemAntiCheater odfiltrowywane są dane testowe z user.ratings,
# ponieważ w metodzie __init__ są one jeszcze dostępne i mogłyby zaburzyć uczciwość konkurencji.
# Jeśli nastąpią zmiany w środowisku testowym i dane te nie będą już zawarte w obiekcie użytkownika w fazie __init__, 
# można użyć rozwiązania z klasy poniżej ("MySystemWithoutAntiCheat"), która różni się tylko tym, że nie odfiltrowuje ręcznie danych testowych.
"""


from RatingSystem import RatingSystem
import numpy as np
import math
import test_users

class MySystemAntiCheater(RatingSystem):
    def __init__(self):
        super().__init__()
        
        self.TOP_K = 15
        self.MIN_OVERLAP = 5.0
        self.BAYES_C = 5.0
        
        self.global_mean = 2.5
        self.movie_bayesian_avgs = {}
        self.user_avgs = {}
        self.user_stdevs = {} 
        self.user_centered_ratings = {}
        self.user_norms = {}
        self.movie_to_users = {} 
        self.user_whole_ratio = {}
        
        self._precompute()

    def _precompute(self):
        all_sum = 0.0
        all_count = 0
        
        # Filtrujemy pary testowe (anticheat)
        blacklisted_pairs = {(int(pair[0]), int(pair[1])) for pair in test_users.test_pairs}
        
        for ratings in self.movie_ratings.values():
            if ratings:
                all_sum += sum(ratings)
                all_count += len(ratings)
        self.global_mean = all_sum / all_count if all_count > 0 else 2.5

        for m_id, ratings in self.movie_ratings.items():
            n = len(ratings)
            if n > 0:
                self.movie_bayesian_avgs[m_id] = (sum(ratings) + self.BAYES_C * self.global_mean) / (n + self.BAYES_C)
        
        for user_id, user_obj in self.users.items():
            clean_ratings = {
                m: r for m, r in user_obj.ratings.items() 
                if (int(user_id), int(m)) not in blacklisted_pairs
            }
            
            if not clean_ratings:
                self.user_whole_ratio[user_id] = 0.5
                self.user_stdevs[user_id] = 1.0 
                self.user_avgs[user_id] = self.global_mean
                continue
                
            avg = sum(clean_ratings.values()) / len(clean_ratings)
            self.user_avgs[user_id] = avg
            
            centered = {m: r - avg for m, r in clean_ratings.items()}
            self.user_centered_ratings[user_id] = centered
            
            variance = sum(c * c for c in centered.values()) / len(centered)
            self.user_stdevs[user_id] = math.sqrt(variance) + 0.15 if variance > 0 else 1.0
            
            self.user_norms[user_id] = math.sqrt(sum(c * c for c in centered.values()))
            
            for m, c in centered.items():
                if m not in self.movie_to_users:
                    self.movie_to_users[m] = {}
                self.movie_to_users[m][user_id] = c

            ratings_list = list(clean_ratings.values())
            whole_count = sum(1 for r in ratings_list if r.is_integer())
            self.user_whole_ratio[user_id] = whole_count / len(ratings_list)

    def rate(self, user, movie_id):
        u_id = getattr(user, 'id', user)
        
        m_bayesian = self.movie_bayesian_avgs.get(movie_id, self.global_mean)
        if u_id in self.user_avgs:
            u_bias = self.user_avgs[u_id] - self.global_mean
            m_bias = m_bayesian - self.global_mean
            fallback_pred = self.global_mean + u_bias + m_bias
        else:
            fallback_pred = m_bayesian
            
        fallback_pred = float(np.clip(fallback_pred, 0.5, 5.0))
        
        if movie_id not in self.movie_to_users or u_id not in self.user_centered_ratings:
            return math.floor(fallback_pred * 2.0 + 0.5) / 2.0
            
        target_avg = self.user_avgs[u_id]
        target_centered = self.user_centered_ratings[u_id]
        target_norm = self.user_norms[u_id]
        target_stdev = self.user_stdevs.get(u_id, 1.0)
        
        if target_norm == 0:
            return math.floor(float(np.clip(target_avg, 0.5, 5.0)) * 2.0 + 0.5) / 2.0
            
        neighbors = self.movie_to_users[movie_id]
        similarities = []
        
        for neighbor_id, neighbor_movie_centered_rating in neighbors.items():
            if neighbor_id == u_id:
                continue
                
            neighbor_norm = self.user_norms.get(neighbor_id, 0)
            if neighbor_norm == 0:
                continue
                
            neighbor_centered = self.user_centered_ratings[neighbor_id]
            dot_product = 0.0
            overlap_count = 0
            
            if len(target_centered) < len(neighbor_centered):
                for m, c in target_centered.items():
                    if m in neighbor_centered:
                        dot_product += c * neighbor_centered[m]
                        overlap_count += 1
            else:
                for m, c in neighbor_centered.items():
                    if m in target_centered:
                        dot_product += target_centered[m] * c
                        overlap_count += 1
                        
            raw_sim = dot_product / (target_norm * neighbor_norm)
            
            amplified_sim = raw_sim * (abs(raw_sim) ** 1.5)
            
            significance_weight = min(overlap_count, self.MIN_OVERLAP) / self.MIN_OVERLAP
            sim = amplified_sim * significance_weight
            
            if sim > 0:
                similarities.append((sim, neighbor_movie_centered_rating, neighbor_id))
                
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:self.TOP_K]
        
        if not top_k:
            prediction = fallback_pred
        else:
            numerator_z = 0.0
            denominator = 0.0
            
            for sim, centered_rating, n_id in top_k:
                n_stdev = self.user_stdevs.get(n_id, 1.0)
                z_score = centered_rating / n_stdev
                numerator_z += sim * z_score
                denominator += sim
                
            if denominator == 0:
                prediction = fallback_pred
            else:
                avg_z = numerator_z / denominator
                prediction = target_avg + (target_stdev * avg_z)
        
        pred = float(np.clip(prediction, 0.5, 5.0))
        whole_ratio = self.user_whole_ratio.get(u_id, 0.5)
        
        if whole_ratio > 0.85:
            return float(math.floor(pred + 0.5))
        elif whole_ratio < 0.15:
            return float(math.floor(pred) + 0.5)
        else:
            return math.floor(pred * 2.0 + 0.54) / 2.0

    def __str__(self):
        return 'System 155294 and 155877 (with anti-cheat filtering)'
    
import numpy as np
import math
from RatingSystem import RatingSystem

class MySystemWithoutAntiCheat(RatingSystem):
    def __init__(self):
        super().__init__()
        
        self.TOP_K = 15
        self.MIN_OVERLAP = 5.0
        self.BAYES_C = 5.0
        
        self.global_mean = 2.5
        self.movie_bayesian_avgs = {}
        self.user_avgs = {}
        self.user_stdevs = {} 
        self.user_centered_ratings = {}
        self.user_norms = {}
        self.movie_to_users = {} 
        self.user_whole_ratio = {}
        
        self._precompute()

    def _precompute(self):
        all_sum = 0.0
        all_count = 0
        
        for ratings in self.movie_ratings.values():
            if ratings:
                all_sum += sum(ratings)
                all_count += len(ratings)
        self.global_mean = all_sum / all_count if all_count > 0 else 2.5

        for m_id, ratings in self.movie_ratings.items():
            n = len(ratings)
            if n > 0:
                self.movie_bayesian_avgs[m_id] = (sum(ratings) + self.BAYES_C * self.global_mean) / (n + self.BAYES_C)
        
        for user_id, user_obj in self.users.items():
            
            clean_ratings = user_obj.ratings
            
            if not clean_ratings:
                self.user_whole_ratio[user_id] = 0.5
                self.user_stdevs[user_id] = 1.0 
                self.user_avgs[user_id] = self.global_mean
                continue
                
            avg = sum(clean_ratings.values()) / len(clean_ratings)
            self.user_avgs[user_id] = avg
            
            centered = {m: r - avg for m, r in clean_ratings.items()}
            self.user_centered_ratings[user_id] = centered
            
            variance = sum(c * c for c in centered.values()) / len(centered)
            self.user_stdevs[user_id] = math.sqrt(variance) + 0.15 if variance > 0 else 1.0
            
            self.user_norms[user_id] = math.sqrt(sum(c * c for c in centered.values()))
            
            for m, c in centered.items():
                if m not in self.movie_to_users:
                    self.movie_to_users[m] = {}
                self.movie_to_users[m][user_id] = c

            ratings_list = list(clean_ratings.values())
            whole_count = sum(1 for r in ratings_list if r.is_integer())
            self.user_whole_ratio[user_id] = whole_count / len(ratings_list)

    def rate(self, user, movie_id):
        u_id = getattr(user, 'id', user)
        
        m_bayesian = self.movie_bayesian_avgs.get(movie_id, self.global_mean)
        if u_id in self.user_avgs:
            u_bias = self.user_avgs[u_id] - self.global_mean
            m_bias = m_bayesian - self.global_mean
            fallback_pred = self.global_mean + u_bias + m_bias
        else:
            fallback_pred = m_bayesian
            
        fallback_pred = float(np.clip(fallback_pred, 0.5, 5.0))
        
        if movie_id not in self.movie_to_users or u_id not in self.user_centered_ratings:
            return math.floor(fallback_pred * 2.0 + 0.5) / 2.0
            
        target_avg = self.user_avgs[u_id]
        target_centered = self.user_centered_ratings[u_id]
        target_norm = self.user_norms[u_id]
        target_stdev = self.user_stdevs.get(u_id, 1.0)
        
        if target_norm == 0:
            return math.floor(float(np.clip(target_avg, 0.5, 5.0)) * 2.0 + 0.5) / 2.0
            
        neighbors = self.movie_to_users[movie_id]
        similarities = []
        
        for neighbor_id, neighbor_movie_centered_rating in neighbors.items():
            if neighbor_id == u_id:
                continue
                
            neighbor_norm = self.user_norms.get(neighbor_id, 0)
            if neighbor_norm == 0:
                continue
                
            neighbor_centered = self.user_centered_ratings[neighbor_id]
            dot_product = 0.0
            overlap_count = 0
            
            if len(target_centered) < len(neighbor_centered):
                for m, c in target_centered.items():
                    if m in neighbor_centered:
                        dot_product += c * neighbor_centered[m]
                        overlap_count += 1
            else:
                for m, c in neighbor_centered.items():
                    if m in target_centered:
                        dot_product += target_centered[m] * c
                        overlap_count += 1
                        
            raw_sim = dot_product / (target_norm * neighbor_norm)
            
            amplified_sim = raw_sim * (abs(raw_sim) ** 1.5)
            
            significance_weight = min(overlap_count, self.MIN_OVERLAP) / self.MIN_OVERLAP
            sim = amplified_sim * significance_weight
            
            if sim > 0:
                similarities.append((sim, neighbor_movie_centered_rating, neighbor_id))
                
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:self.TOP_K]
        
        if not top_k:
            prediction = fallback_pred
        else:
            numerator_z = 0.0
            denominator = 0.0
            
            for sim, centered_rating, n_id in top_k:
                n_stdev = self.user_stdevs.get(n_id, 1.0)
                z_score = centered_rating / n_stdev
                numerator_z += sim * z_score
                denominator += sim
                
            if denominator == 0:
                prediction = fallback_pred
            else:
                avg_z = numerator_z / denominator
                prediction = target_avg + (target_stdev * avg_z)
        
        pred = float(np.clip(prediction, 0.5, 5.0))
        whole_ratio = self.user_whole_ratio.get(u_id, 0.5)
        
        if whole_ratio > 0.85:
            return float(math.floor(pred + 0.5))
        elif whole_ratio < 0.15:
            return float(math.floor(pred) + 0.5)
        else:
            return math.floor(pred * 2.0 + 0.54) / 2.0

    def __str__(self):
        return 'MySystem 155294 and 155877 (no anti-cheat filtering)'