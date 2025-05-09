from tqdm import tqdm
from RatingLib import User
from RatingLib import Movie
import test_users
import numpy as np
import copy
test_users = test_users.test_users

class RatingSystem:
    """ 
        Object of this class represents a single rating system. It does not use abc library, however it is implied, that this class should be used as if it was an abstract class.
    """
    def __init__(self):
        """
            Default system initializer - it loads up users and movie ratings into the self.user and self.movie_ratings indices, which are implemented as (dictionary). If you are willing to use the indices as ndarrays, please perform the conversion in the specific (non-abstract) system definition. 
            The indices are filled with use of User.index, which is a class parameter, and User.ratings, which is an object parameter. The initializer also makes use of the 'test_users.py' file, which should contain a single definition of a test_users array, which consists of user identifiers to be ignored by the system. Be ware, that those indices should be filled prior to initializing a new system
        """
        self.users = {id : User.index[id] for id in User.index if id not in test_users}
        self.movie_ratings = {}
        for user in tqdm(self.users):
            for movie in self.users[user].ratings:
                if movie not in self.movie_ratings.keys():
                    self.movie_ratings[movie] = [self.users[user].ratings[movie]]
                else:
                    self.movie_ratings[movie].append(self.users[user].ratings[movie])
        
    def rate(self, user : User, movie : Movie):
        """
            Altough we are not using abc in the implementation, this method is to be interpreted as an abstract method (or pure virtual if you come from the C++ community).
            In reality this is an empty method stub, which returns None. This method is supposed to be overrided by your system implementation.
            The implementation of this method should return a movie rating for a given user, the competition manager is supposed to make sure that given user did not grade a given movie.
            
            :param user: User object (User)
            :param movie: Movie object (Movie)
        """
        return
    
class RatingSystemCompetition:
    """
        This is a Rating System manager. Object of the manager is supposed to run the competition between rating systems based on a classic round robin strategy.
    """
    def __init__(self, verbose : int = 2):
        """
            The indices are filled with use of User.index, which is a class parameter. The initializer also makes use of the 'test_users.py' file, which should contain a single definition of a test_users array, which consists of user identifiers to be ignored by the system. Be ware, that those indices should be filled prior to initializing a new system.

            The initializers implements object parameter verbose, which controls the output to the stdio of the manager object. 
            0 - print nothing
            1 - print the final scores along with MSE and MAE of every system and everything specified in the previous levels of verbose
            2 (default) - print specific matchup details and everything specified in the previous levels of verbose

            :param verbose: verbosity of the rating system manager (numerical, default = 2)

        """
        self.registered_systems = []
        self.users = {id : User.index[id] for id in User.index if id not in test_users}
        self.verbose = verbose
        self.errors = {}
    def register(self, system : RatingSystem):
        """
            register a given system to the competition

            :param system: System object (RatingSystem)
        """
        self.registered_systems.append(system)
        self.errors[str(system)] = []
    def mse(self, system : str):
        """
            calculate Root of the Mean Squared Error (RMSE) of the errors generated by this system in the competition

            :param system: Name of the evaluated system (str)
            :returns: RMSE
        """
        errors = np.array(self.errors[system])
        return np.sqrt(np.sum(np.power(errors, 2)) / errors.size)
    def mae(self, system : str):
        
        """
            calculate Mean Absolute Error (MAE) of the errors generated by this system in the competition

            :param system: Name of the evaluated system (str)
            :returns: MAE
        """
        errors = np.array(self.errors[system])
        return np.sum(np.abs(errors))/errors.size
    def build_round_robin(self):
        """
            Note this method has to (it is obligatory to call it, otherwise competition will consist of no matchups) be called prior to the competition. This method builds all matchups in the competition. 
        """
        self.pairs = {}
        for system in self.registered_systems:
            self.pairs[system]  = []
            for competitor in self.registered_systems:
                if str(system) != str(competitor):
                    self.pairs[system].append((system, competitor))

            
    def runMatch(self, system : RatingSystem, competitor: RatingSystem, sample_size : int = 100):
        """
            This methods runs a match between two given systems. A sample of a 'sample_size' (default 100) Users is drawn from all test users. For each user a copy is created. The newly created copy does not contain a single movie, which was originally rated by this user. Systems are asked to predict the rating for this specific user and this specific movie. The system with lower absolute error for that user is deemed a winner. 

            :param system: evaluated system (RatingSystem)
            :param competitor: competitor system (RatingSystem)

            :returns: a tuple (wins_by_system(int), draws(int), wins_by_competitor(int))
        """
        users_ids = np.random.choice(np.array(list(self.users.keys())), sample_size)
        score = 0
        wins = 0
        loses = 0
        draws = 0
        for user_id in users_ids:
            user = self.users[user_id]
            user_copy = copy.deepcopy(self.users[user_id])
            movie_id = np.random.choice(np.array(list(user.ratings.keys())), size=1)[0]
            del user_copy.ratings[movie_id]
            true_rating = self.users[user_id].ratings[movie_id]
            system_rating = system.rate(user_copy,movie_id)
            competitor_rating = competitor.rate(user_copy,movie_id)
            self.errors[str(system)].append(true_rating - system_rating)
            self.errors[str(competitor)].append(true_rating - competitor_rating)
            if abs(true_rating - system_rating) <  abs(true_rating - competitor_rating):
                score += 1
                wins += 1
            elif abs(true_rating - system_rating) >  abs(true_rating - competitor_rating):
                score -= 1
                loses += 1
            else:
                draws += 1
                
        return score, wins, draws, loses
    
    def compete(self):
        """
            Run the competition and print out the results based on the verbose parameter. Note, the only output is fed to the stdio, currently the system does not provide evaluation in any other form. 
        """
        self.total_scores = {}
        for system in self.pairs:
            self.total_scores[system] = 0
            if self.verbose >= 2: print(f'{system} analysis: ')
            for matchup in self.pairs[system]:
                score, wins, draws, loses = self.runMatch(matchup[0],matchup[1])
                if self.verbose >= 2: print(f'{matchup[0]} vs {matchup[1]} : {score} ({wins} wins, {draws} draws, {loses} loses)')
                self.total_scores[system] += score
            if self.verbose >= 2: print(f'{system} score: {self.total_scores[system]}')
        if self.verbose >= 1:
            print('Final scores: ')
            place = 1
            for system in sorted(self.total_scores, key=self.total_scores.get, reverse=True):
                print(f'{place}. {system}, {self.total_scores[system]} pkt, RMSE: {self.mse(str(system))}, MAE: {self.mae(str(system))}')
                place += 1
            
        
            
            
            
            
    