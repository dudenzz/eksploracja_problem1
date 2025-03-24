from tqdm import tqdm
from RatingLib import User
from RatingLib import Movie
import test_users
import numpy as np
test_users = test_users.test_users

class RatingSystem:
    def __init__(self):
        self.users = {id : User.index[id] for id in User.index if id not in test_users}
        self.movie_ratings = {}
        for user in tqdm(self.users):
            for movie in self.users[user].ratings:
                if movie not in self.movie_ratings.keys():
                    self.movie_ratings[movie] = [self.users[user].ratings[movie]]
                else:
                    self.movie_ratings[movie].append(self.users[user].ratings[movie])
        
    def rate(self, user, movie):
        return
    
import copy
class RatingSystemCompetition:
    
    def __init__(self):
        self.registered_systems = []
        self.users = {id : User.index[id] for id in User.index if id not in test_users}
        self.verbose = 2
    def register(self, system):
        self.registered_systems.append(system)
        
    def build_round_robin(self):
        self.pairs = {}
        for system in self.registered_systems:
            self.pairs[system]  = []
            for competitor in self.registered_systems:
                if str(system) != str(competitor):
                    self.pairs[system].append((system, competitor))

            
    def runMatch(self, system, competitor):
        users_ids = np.random.choice(np.array(list(self.users.keys())), size=100)
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
                print(f'{place}. {system}, {self.total_scores[system]} pkt')
                place += 1
            
        
            
            
            
            
    