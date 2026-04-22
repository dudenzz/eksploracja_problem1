import csv
from RatingSystem import RatingSystem
from RatingLib import User, Movie

class MySystem(RatingSystem):
    def __init__(self):
        super().__init__()
        self.mu = 0.0
        self.b_u = {}
        self.b_i = {}
        
        self.reg_u = 15.0
        self.reg_i = 5.0

        self._load_genres()
        
        self._train_baseline()

    def _load_genres(self):
        try:
            with open('../data/movie.csv', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for line in csv_reader:
                    if len(line) >= 3:
                        movie_id = int(line[0])
                        genres_str = line[2]
                        genres_list = genres_str.split('|')
                        
                        if movie_id in Movie.index:
                            Movie.index[movie_id].genres = genres_list
        except Exception as e:
            print(f"Error while loading genres in MySystem: {e}")

    def _train_baseline(self):
        total_rating = 0
        total_count = 0
        
        for movie_id, ratings in self.movie_ratings.items():
            total_rating += sum(ratings)
            total_count += len(ratings)
            
        if total_count > 0:
            self.mu = total_rating / total_count
        else:
            self.mu = 2.5
            
        for movie_id, ratings in self.movie_ratings.items():
            if len(ratings) > 0:
                dev_sum = sum([r - self.mu for r in ratings])
                self.b_i[movie_id] = dev_sum / (len(ratings) + self.reg_i)
            else:
                self.b_i[movie_id] = 0.0
                
        for user_id, user_obj in User.index.items():
            dev_sum = 0
            count = 0
            for movie_id, rating in user_obj.ratings.items():
                b_i_val = self.b_i.get(movie_id, 0.0)
                dev_sum += (rating - self.mu - b_i_val)
                count += 1
                
            if count > 0:
                self.b_u[user_id] = dev_sum / (count + self.reg_u)
            else:
                self.b_u[user_id] = 0.0

    def rate(self, user, movie):
        if movie in user.ratings:
            return user.ratings[movie]
            
        user_bias = self.b_u.get(user.id, 0.0)
        movie_bias = self.b_i.get(movie, 0.0)
        baseline_pred = self.mu + user_bias + movie_bias
        
        target_movie_obj = Movie.index.get(movie)
        
        if not target_movie_obj or not target_movie_obj.genres:
            return self._clip_rating(baseline_pred)
            
        target_genres = set(target_movie_obj.genres)
        
        sim_sum = 0.0
        weighted_diff_sum = 0.0
        
        for rated_movie_id, actual_rating in user.ratings.items():
            rated_movie_obj = Movie.index.get(rated_movie_id)
            
            if not rated_movie_obj or not rated_movie_obj.genres:
                continue
                
            rated_genres = set(rated_movie_obj.genres)
            if not rated_genres or rated_genres == {"(no genres listed)"}:
                continue
                
            intersection = len(target_genres.intersection(rated_genres))
            union = len(target_genres.union(rated_genres))
            sim = intersection / union if union > 0 else 0.0
            
            if sim > 0:
                rated_movie_bias = self.b_i.get(rated_movie_id, 0.0)
                rated_baseline = self.mu + user_bias + rated_movie_bias
                residual = actual_rating - rated_baseline
                
                weighted_diff_sum += sim * residual
                sim_sum += sim
                
        if sim_sum > 0:
            adjustment = weighted_diff_sum / sim_sum
            final_pred = baseline_pred + adjustment
        else:
            final_pred = baseline_pred
            
        return self._clip_rating(final_pred)

    def _clip_rating(self, prediction):
        if prediction > 5.0:
            return 5.0
        if prediction < 1.0:
            return 1.0
        return prediction

    def __str__(self):
        return 'System created by 156962 and 155994'