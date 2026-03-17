"""
Bazowy moduł oceny dla filmów i użytkowników
"""
from RatingSystem import RatingSystem

class BiasRatingSystem(RatingSystem):
    """
    Klasa reprezentująca działanie Bias Rating System
    """
    def __init__(self):
        super().__init__()
        # parametr regularyzacji występuje w mianowniku
        # w celu obniżenia biasu przy małej liczbie oceny użytkownika
        self.lambda_r = 10
        self.mean_global = self.calculate_global_mean()

    def calculate_global_mean(self):
        """
        Ta metoda zwraca globalną średnią ocen wszystkich filmów.
        """
        total = 0
        count = 0

        for ratings in self.movie_ratings.values():
            total += sum(ratings)
            count += len(ratings)

        if count == 0:
            return 2.5

        return total / count
    def calculate_user_bias(self, user):
        """
        Ta metoda zwraca bias użytkownika,
        jego wskaźnik podatności na ocenę + (pozytywnie) - (negatywnie).
        """
        ratings = user.ratings.values()
        n = len(ratings)

        if n == 0:
            return 0

        diff_sum = sum(r - self.mean_global for r in ratings)

        return diff_sum / (self.lambda_r + n)
    def calculate_movie_bias(self, movie):
        """
        Ta metoda zwraca bias filmu czyli jego wskaźnik ocen + (pozytywnie) - (negatywnie).
        """
        if movie not in self.movie_ratings:
            return 0

        ratings = self.movie_ratings[movie]
        n = len(ratings)

        if n == 0:
            return 0

        diff_sum = sum(r - self.mean_global for r in ratings)

        return diff_sum / (self.lambda_r + n)
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
        if movie in user.ratings:
            return user.ratings[movie]
        b_u = self.calculate_user_bias(user)
        b_m = self.calculate_movie_bias(movie)

        prediction = self.mean_global + b_u + b_m
        return self.clamp_rating(prediction)
    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 155198 and 155921'
