import csv
from RatingLib import User, Movie
from tqdm import tqdm
from RatingSystem import RatingSystem, RatingSystemCompetition
from SampleSystems import NaiveRating, AverageMovieRating, GlobalAverageMovieRating, Cheater, AverageUserRating
from system156962 import MySystem as System156962
from system155198 import BiasRatingSystem
from system155978 import HybridSystem
from system155864 import System155864
from system156145  import MySystem as System156145  
from system155922_155944 import MySystem as System155922_155944
from system158058 import MySystem as System158058   
from system156027 import MySystem as System156027
from system156014 import MySystem as System156014
from system155898 import MySystem as System155898
from system155956 import MySystem as System155956
from system155294 import MySystemWithoutAntiCheat as System155294_1
from system155294 import MySystemAntiCheater as System155294_2
from system156007 import MySystem as System156007
from system155093 import MySystem as System155093   
from system155974 import MySystem as System155974   
from time import time
# from secret_system import MySystem as secret

def main():
    #read the movie indices
    with open('../data/movie.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in csv_reader:
            Movie(int(line[0]), line[1],line[2].split('|'))
    #read the genome scores
    last_movie = None
    with open('../data/genome_scores.csv', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for line in tqdm(csv_reader, total=11709768):
            movie_id = int(line[0])
            if last_movie == movie_id:
                if movie_id in Movie.index:
                    Movie.index[movie_id].add_genome_score(int(line[1]), float(line[2]))
            elif last_movie:
                if movie_id in Movie.index:
                    Movie.index[last_movie].sort_tags()
            last_movie = movie_id
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
    # competition.register(secret())
    competition.register(NaiveRating())
    competition.register(AverageMovieRating())
    competition.register(GlobalAverageMovieRating())
    competition.register(Cheater())
    competition.register(AverageUserRating())
    total_start = time()
    print('registering student systems...')
    systems = []
    times = []
    print('registering BiasRatingSystem...')
    start = time()
    competition.register(BiasRatingSystem())
    end = time()
    systems.append('BiasRatingSystem')
    times.append(end - start)
    print(f'BiasRatingSystem registered in {end - start:.2f} seconds')
    
    print('registering HybridSystem...')
    start = time()
    competition.register(HybridSystem())
    end = time()
    systems.append('HybridSystem')
    times.append(end - start)
    print(f'HybridSystem registered in {end - start:.2f} seconds')
    
    print('registering System155922_155944...')
    start = time()
    competition.register(System155922_155944())
    end = time()
    systems.append('System155922_155944')
    times.append(end - start)
    print(f'System155922_155944 registered in {end - start:.2f} seconds')
    
    print('registering System156145...')
    start = time()
    competition.register(System156145())
    end = time()
    systems.append('System156145')
    times.append(end - start)
    print(f'System156145 registered in {end - start:.2f} seconds')
    print('registering System155864...')
    start = time()
    competition.register(System155864())
    end = time()
    systems.append('System155864')
    times.append(end - start)
    print(f'System155864 registered in {end - start:.2f} seconds')
    print('registering System158058...' )
    start = time()
    competition.register(System158058())
    end = time()
    systems.append('System158058')
    times.append(end - start)
    print(f'System158058 registered in {end - start:.2f} seconds')
    
    print('registering System156962...')
    start = time()
    competition.register(System156962())
    end = time()
    systems.append('System156962')
    times.append(end - start)

    print(f'System156962 registered in {end - start:.2f} seconds')
    print('registering System156027...')
    start = time()
    competition.register(System156027())
    end = time()
    systems.append('System156027')
    times.append(end - start)
    print(f'System156027 registered in {end - start:.2f} seconds')
    print('registering System156014...')
    start = time()
    competition.register(System156014())
    end = time()
    systems.append('System156014')
    times.append(end - start)
    print(f'System156014 registered in {end - start:.2f} seconds')

    print('registering System155898...')
    start = time()
    competition.register(System155898())
    end = time()
    systems.append('System155898')
    times.append(end - start)
    print(f'System155898 registered in {end - start:.2f} seconds')
    print('registering System155956...')
    start = time()
    competition.register(System155956())
    end = time()
    systems.append('System155956')
    times.append(end - start)
    print(f'System155956 registered in {end - start:.2f} seconds')
    print('registering System155294_1 (no anti-cheat filtering)...')
    start = time()
    competition.register(System155294_1())
    end = time()
    systems.append('System155294_1')
    times.append(end - start)
    print(f'System155294_1 registered in {end - start:.2f} seconds')
    print('registering System155294_2 (with anti-cheat filtering)...')
    start = time()
    competition.register(System155294_2())
    end = time()
    systems.append('System155294_2')
    times.append(end - start)
    print(f'System155294_2 registered in {end - start:.2f} seconds')
    print('registering System156007...')
    start = time()
    competition.register(System156007())
    end = time()
    systems.append('System156007')
    times.append(end - start)
    print(f'System156007 registered in {end - start:.2f} seconds')
    print('registering System155093...')
    start = time()
    competition.register(System155093())
    end = time()
    systems.append('System155093')
    times.append(end - start)
    print(f'System155093 registered in {end - start:.2f} seconds')
    print('registering System155974...')
    start = time()
    competition.register(System155974())  
    end = time()
    systems.append('System155974')
    times.append(end - start)
    print(f'System155974 registered in {end - start:.2f} seconds')
    total_end = time()
    print(f'Total registration time: {total_end - total_start:.2f} seconds')
    print('Registration time table:')
    print('System Name\tRegistration Time (seconds)')
    for name, t in zip(systems, times):
        print(f'{name}\t{t:.2f}')

    print('All systems registered. Building round robin schedule...')
    for i in range(10):
        start = time()
        print('--------------------------------------------------------')
        print(f'Building round robin schedule: iteration {i+1}/10')
        competition.build_round_robin()
        #run the competition - it prints out the results
        print(f'results after round robin {i+1}/10:')
        competition.compete()
        end = time()
        print(f'Round robin {i+1} completed in {end - start:.2f} seconds')

if __name__ == "__main__":
    main()