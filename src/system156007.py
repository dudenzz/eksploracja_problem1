from RatingSystem import RatingSystem
import numpy as np
from RatingLib import Movie
class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.movies = {id : Movie.index[id] for id in Movie.index}
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
        """
        Ta metoda zwraca rating w skali 1-5. Jest to ocena przyznana przez użytkownika 'user' filmowi 'movie'.
        """
        movie_avg=self.calc_movie_avg(movie)
        # user_avg=self.calc_user_avg(user)
        rate_num = len(user.ratings)
        movies_rated_by_user = list(user.ratings.keys())
        
        random_sample = np.random.choice(movies_rated_by_user, 100, replace=False) if rate_num >= 100 else np.random.choice(movies_rated_by_user, rate_num, replace=False)
        offset = 0
        for movie in random_sample:
            curr_movie = self.calc_movie_avg(movie)
            offset += curr_movie-user.ratings[movie]
        offset/=100
        return movie_avg-offset

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 156007 and 155833'
