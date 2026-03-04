from RatingSystem import RatingSystem

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        """
        Ta metoda zwraca rating w skali 1-5. Jest to ocena przyznana przez użytkownika 'user' filmowi 'movie'.
        """
        if movie in user.ratings:
            return user.ratings[movie]
        else:
            return 2.5
    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return 'System created by 111333 and 333444'
