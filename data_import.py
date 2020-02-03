# importing libaries
import numpy as np
import pandas as pd

import requests, json, csv
from time import sleep

# making a batch of api requests to the imdb website for extra movie features
def api_request(imdb_ids, api_key, file_name, sleep_time):
    request_file = open(file_name, 'w')
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
        # appending movie features onto csv file
        csv_writer.writerow([imdb_id, age_rating, genres, director, actors, plot])
        # adding sleep between requests to avoid hitting request limit
        sleep(sleep_time)
    request_file.close()
    return print("file finished")