from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.global_mean = 0
        self.user_bias = {}
        self.movie_bias = {}
        
        if len(self.movie_ratings) > 0:
            all_ratings = []
            for movie_id, ratings in self.movie_ratings.items():
                all_ratings.extend(ratings)
            self.global_mean = np.mean(all_ratings) if all_ratings else 2.5
            
            for user_id, user_obj in self.users.items():
                if len(user_obj.ratings) > 0:
                    self.user_bias[user_id] = np.mean(list(user_obj.ratings.values())) - self.global_mean
                else:
                    self.user_bias[user_id] = 0
                    
            for movie_id, ratings in self.movie_ratings.items():
                if len(ratings) > 0:
                    self.movie_bias[movie_id] = np.mean(ratings) - self.global_mean
                else:
                    self.movie_bias[movie_id] = 0

    def rate(self, user, movie):
        movie_id = movie
        user_id = user.id
        
        user_bias = self.user_bias.get(user_id, 0)
        movie_bias = self.movie_bias.get(movie_id, 0)
        
        prediction = self.global_mean + user_bias + movie_bias
        
        return np.clip(prediction, 1.0, 5.0)
    
    def __str__(self):
        return 'System created by 155093'