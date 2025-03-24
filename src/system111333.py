from RatingSystem import RatingSystem

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()

    def rate(self, user, movie):
        if movie in user.ratings:
            return user.ratings[movie]
        else:
            return 2.5
    def __str__(self):
        return 'System created by 111333'
