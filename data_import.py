# importing libaries
import numpy as np
import pandas as pd

import requests, json, csv
from time import sleep

# making a series of api requests to the OMDb database for extra movie features
def api_request(imdb_ids, api_key, output_file, sleep_time):
    # writing output file
    request_file = open(output_file, 'w')
    csv_writer = csv.writer(request_file)
    csv_writer.writerow(['imdb_id', 'age_rating', 'genres', 'director', 'actors', 'plot'])
    # looping over movies
    for imdb_id in imdb_ids:
        try:
            # making api request for movie
            request = requests.get("http://www.omdbapi.com/?apikey={}&i={}".format(api_key, imdb_id))
            # converting request from json format into dictionary
            response = request.json()
        except:
            # exception handling for unsuccessful requests
            continue
        # extracting important features from request
        age_rating = response.get('Rated', np.nan)
        genres = response.get('Genre', np.nan)
        director = response.get('Director', np.nan)
        actors = response.get('Actors', np.nan)
        plot = response.get('Plot', np.nan)
        # appending movie features to csv file
        csv_writer.writerow([imdb_id, age_rating, genres, director, actors, plot])
        # adding sleep between requests to avoid hitting request rate limit
        sleep(sleep_time)
    request_file.close()
    return print("file finished")

# merging movies with requests
def movie_merger(movies_file, links_file, requests_file, merged_file):
    # importing movies, links and requests
    movies_df = pd.read_csv(movies_file)
    links_df = pd.read_csv(links_file)
    requests_df = pd.read_csv(requests_file).drop(columns = ['Unnamed: 0']).drop_duplicates()
    # merging imdb links with movies
    movies_df = pd.merge(movies_df, links_df, left_on = "movieId", right_on = "movieId").drop(columns = ["tmdbId"])
    # reformatting imdb ids
    movies_df['imdbId'] = movies_df['imdbId'].map(lambda x: 'tt{0:07d}'.format(x))
    # merging requests with movies
    movies_df = pd.merge(movies_df, requests_df, left_on = 'imdbId', right_on = 'imdb_id').drop(columns = ['genres_x', 'imdbId', 'imdb_id'])
    # renaming columns after merge
    movies_df['genres'] = movies_df['genres_y']
    movies_df['ageRating'] = movies_df['age_rating']
    movies_df.drop(columns = ['genres_y', 'age_rating'], inplace = True)
    # saving merged dataframe to a csv file
    movies_df.to_csv(merged_file)
    return print("files merged")
