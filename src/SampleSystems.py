from RatingSystem import RatingSystem

class NaiveRating(RatingSystem):
    def __init__(self):
        super().__init__()
    def rate(self, user, movie):
        return 2.5
    def __str__(self):
        return 'Naive Rating'

class AverageMovieRating(RatingSystem):
    def __init__(self):
        super().__init__()
    def rate(self, user, movie):
        n = len(self.movie_ratings[movie])
        if n == 0:
            return 2.5
        else:
            return sum(self.movie_ratings[movie])/n
    def __str__(self):
        return 'Average Movie Rating'
class AverageUserRating(RatingSystem):
    def __init__(self):
        super().__init__()
    def rate(self, user, movie):
        n = len(user.ratings.values())
        if n == 0:
            return 2.5
        else:
            return sum(user.ratings.values())/n
    def __str__(self):
        return 'Average User Rating'

class GlobalAverageMovieRating(RatingSystem):
    def __init__(self):
        super().__init__()
        self.GlobalAverageMovieRating = 0
        self.TotalMovies = 0
        for movie in self.movie_ratings:
            for rating  in self.movie_ratings[movie]:
                self.GlobalAverageMovieRating += rating
                self.TotalMovies += 1
        self.GlobalAverageMovieRating /= self.TotalMovies

    def rate(self, user, movie):
        return self.GlobalAverageMovieRating
    def __str__(self):
        return 'Average Global Movie Rating'
    
class Cheater(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        if movie in user.ratings:
            return user.ratings[movie]
        else:
            return 2.5
    def __str__(self):
        return 'Cheater'
