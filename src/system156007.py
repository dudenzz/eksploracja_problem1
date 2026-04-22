from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie
import copy

# This system requires additional information from file genome_scores.csv. Required changes in run.py and RatingLib.py are implemented in pr: https://github.com/dudenzz/eksploracja_problem1/pull/8

class MySystem(RatingSystem):
    def __init__(self, num_tags=10, weight=0.1, threshold = 0.05):
        super().__init__()
        self.tags=num_tags
        self.w=weight
        self.t=threshold
        self.movies = {id : Movie.index[id] for id in Movie.index}
    def jaccard_similarity(self, list_a, list_b) -> float:
        set_a = set(list_a)
        set_b = set(list_b)
        if not set_a and not set_b:
            return 1.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    def calc_movie_avg(self,movie):
        n = len(self.movie_ratings[movie])
        if n == 0:
            return 2.5
        else:
            return sum(self.movie_ratings[movie])/n
    def calc_user_avg(self,user):
        n = len(user.ratings.values())
        if n == 0:
            return 2.5
        else:
            return sum(user.ratings.values())/n
    def rate(self, user, movie):
        num = 0
        weighted_sum = 0
        similarity_sum = 0
        
        target_movie = Movie.index[movie]
        target_tags = target_movie.sorted_tags[:self.tags]

        for rated_movie_id, user_rating in user.ratings.items():
            comparison_movie = Movie.index[rated_movie_id]
            jacc_sim_gen = self.jaccard_similarity(target_movie.genres, comparison_movie.genres)
            comp_tags = comparison_movie.sorted_tags[:self.tags]
            jacc_sim_tag = self.jaccard_similarity(target_tags, comp_tags)
            
            full_sim = (jacc_sim_gen * self.w + jacc_sim_tag * (1 - self.w))
            
            if full_sim >= self.t:
                weighted_sum += (user_rating * full_sim)
                similarity_sum += full_sim
                num += 1

        if similarity_sum == 0:
            return self.calc_movie_avg(movie)
            
        return min(max(weighted_sum / similarity_sum, 0.5), 5)
    
    def cross_validate(self, k=5, sample_size=5000):
        all_interactions = []
        for user_id, user_obj in self.users.items():
            for movie_id, rating in user_obj.ratings.items():
                all_interactions.append((user_id, movie_id, rating))
                
        sample_size = min(sample_size, len(all_interactions))
        indices = np.random.choice(len(all_interactions), sample_size, replace=False)
        cv_sample = [all_interactions[i] for i in indices]
        
        fold_size = len(cv_sample) // k
        rmse_scores = []
        
        for fold in range(k):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold != k-1 else len(cv_sample)
            test_set = cv_sample[test_start:test_end]
            
            errors = []
            for user_id, movie_id, true_rating in test_set:
                user_copy = copy.deepcopy(self.users[user_id])
                del user_copy.ratings[movie_id]
                
                pred_rating = self.rate(user_copy, movie_id)
                
                if pred_rating is None or np.isnan(pred_rating):
                    pred_rating = 2.5 
                    
                errors.append((true_rating - pred_rating)**2)
                
            fold_rmse = np.sqrt(np.mean(errors))
            rmse_scores.append(fold_rmse)
            
        mean_rmse = np.mean(rmse_scores)
        
        return mean_rmse
    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 156007 and 155833'