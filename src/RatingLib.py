class Movie:
    index = {}
    name_index = {}
    inner_index = {}
    reverse_inner_index = {}
    inner_index_gen = 0
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.ratings = []
        self.genres = []
        Movie.index[id] = self
        Movie.name_index[name] = self
    def add_rating(self, rating):
        self.ratings.append(rating)
    
    
class User:
    index = {}
    def __init__(self, id):
        self.id = id
        self.ratings = {}
        User.index[id] = self
    def add_rating(self, movie, rating):
        movie.add_rating(rating)
        self.ratings[movie.id] = rating
    def __str__(self):
        str_bldr = f'{self.id}'
        return str_bldr

        
