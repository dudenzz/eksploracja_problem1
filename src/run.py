import csv
from RatingLib import User, Movie
from tqdm import tqdm
from RatingSystem import RatingSystem, RatingSystemCompetition
from SampleSystems import NaiveRating, AverageMovieRating, GlobalAverageMovieRating, Cheater, AverageUserRating
from system111333 import MySystem

def main():
    #read the movie indices
    with open('../data/movie.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in csv_reader:
            Movie(int(line[0]), line[1])
    #read the user indices
    with open('../data/rating.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in tqdm(csv_reader, total=20000263):
            if not int(line[0]) in User.index.keys():
                User(int(line[0]))
            User.index[int(line[0])].add_rating(Movie.index[int(line[1])],float(line[2]))
    
    #create the competition
    competition = RatingSystemCompetition()
    #register systems
    competition.register(MySystem())
    competition.register(NaiveRating())
    competition.register(AverageMovieRating())
    competition.register(GlobalAverageMovieRating())
    competition.register(Cheater())
    competition.register(AverageUserRating())
    #build the matchup table
    competition.build_round_robin()
    #run the competition - it prints out the results
    competition.compete()
       
if __name__ == "__main__":
    main()