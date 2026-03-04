from RatingSystem import RatingSystem

class NaiveRating(RatingSystem):
    """
    Przykładowy system - naiwny. 
    Hipoteza: jeżeli zwrócę każdemu filmowi średnią ocenę (2.5/5), to moja ocena będzie niezła.
    """
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
        """
        Przykładowy system - średnia.
        Hipoteza: jeżeli zwrocę każdeu filmowi średnią ocenę (wynikającą z wszystkich ocen), to moja ocena będzie niezła.
        Jeżeli ten film jeszcze nie był oceniony, to zwrócę 2.5.
        """
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
        """
        Przykładowy system - średnia użytkownika.
        Hipoteza: jeżeli zwrócę dla tego filmu średnią ocenę wystawioną przez użytkownika, to mój system będzie niezły.

        """
        n = len(user.ratings.values())
        if n == 0:
            return 2.5
        else:
            return sum(user.ratings.values())/n
    def __str__(self):
        return 'Average User Rating'

class GlobalAverageMovieRating(RatingSystem):
    def __init__(self):
        """
        Przykładowy system - średnia ocena filmu.
        Hipoteza: średnia ocena tego filmu wśród wszystkich użytkowników powinna być dobrą estymacją.
        """
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
        """
        Testowy system.
        Jeżeli ten system działa, to coś jest nie tak - systemy mają dostęp do ocen filmów, które mają wyznaczyć - powinien działać mniej więcej tak samo jak system naiwny.
        """
        if movie in user.ratings:
            return user.ratings[movie]
        else:
            return 2.5
    def __str__(self):
        return 'Cheater'
