from RatingSystem import RatingSystem


class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        # tutaj wczytać dodatkowe dane

    def rate(self, user, movie):
        """
        Ta metoda zwraca rating w skali 1-5. Jest to ocena przyznana przez użytkownika 'user' filmowi 'movie'.
        """
        n = len(user.ratings.values())
        if n == 0:
            return 3
        else:
            return sum(user.ratings.values()) / (n - 1)
        # tutaj usprawnić działanie 

    def __str__(self):
        """
        Ta metoda zwraca numery indeksów wszystkich twórców rozwiązania. Poniżej przykład.
        """
        return "System created by 155898 and 000000 and 000000"
        # tutaj uzupełnić indeksy
