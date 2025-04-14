class Movie:
    """
        Object of this class represents a single movie
    """
    index = {}
    name_index = {}
    inner_index = {}
    reverse_inner_index = {}
    inner_index_gen = 0
    def __init__(self, id, name):
        """
            Initializer of the movie creates an empty object with a given ID and a name. The movie is indexed in index via index[id] = self and in name_index via name_index[name] = self. There is no reverse index.
            
            :param id: movie identifier (numericial)
            :param name: movie name

        """
        self.id = id
        self.name = name
        self.ratings = []
        self.genres = []
        Movie.index[id] = self
        Movie.name_index[name] = self
    def add_rating(self, rating):
        """
            This method adds a single numerical rating to a movie.

            :param rating: movie rating (numerical)
        """
        self.ratings.append(rating)
    
    
class User:
    """
        Object of this class represents a single user.
    """
    index = {}
    def __init__(self, id):
        """
            Initializer creates an empty User object. User is stored in the user index via index[id] = self. No reverse index is created.

            :param id: user identifier (numerical)
        """
        self.id = id
        self.ratings = {}
        User.index[id] = self
    def add_rating(self, movie, rating):
        """
            This method adds a single numerical rating to a movie, which has been given by this user.

            :param movie: movie identifier (numerical)
            :param rating: movie rating (numerical)
        """
        movie.add_rating(rating)
        self.ratings[movie.id] = rating
    def __str__(self):
        """
            The default to string method. It returns a string value of the user identifier.
        """
        str_bldr = f'{self.id}'
        return str_bldr

        
