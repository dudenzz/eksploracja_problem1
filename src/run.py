import csv
from RatingLib import User, Movie
from tqdm import tqdm
from RatingSystem import RatingSystem, RatingSystemCompetition
from SampleSystems import NaiveRating, AverageMovieRating, GlobalAverageMovieRating, Cheater, AverageUserRating
from system156007 import MySystem
# from StudentSystems import Sys147715
# from secret_system import MySystem as secret

def main():
    #read the movie indices
    with open('./data/movie.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in csv_reader:
            Movie(int(line[0]), line[1])
            
    #read the genome scores
    with open('./data/genome_scores.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in tqdm(csv_reader, total=11709768):
            movie_id = int(line[0])
            if movie_id in Movie.index:
                Movie.index[movie_id].add_genome_score(int(line[1]), float(line[2]))

    #read the user indices
    with open('./data/rating.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in tqdm(csv_reader, total=20000263):
            if not int(line[0]) in User.index.keys():
                User(int(line[0]))
            User.index[int(line[0])].add_rating(Movie.index[int(line[1])],float(line[2]))
    
    #create the competition
    competition = RatingSystemCompetition()
    #register systems
    # competition.register(secret())
    competition.register(NaiveRating())
    competition.register(AverageMovieRating())
    competition.register(GlobalAverageMovieRating())
    competition.register(Cheater())
    competition.register(AverageUserRating())
    competition.register(MySystem())
    competition.build_round_robin()
    #run the competition - it prints out the results
    competition.compete()

if __name__ == "__main__":
    main()