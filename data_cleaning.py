# importing libaries
import numpy as np
import pandas as pd

import re
from sklearn.feature_extraction.text import CountVectorizer

# cleaning movies and ratings
def data_cleaner(movies_file, ratings_file, cleaned_movies_file, cleaned_ratings_file):
    # importing movies and ratings
    movies_df = pd.read_csv(movies_file).drop(columns = ['Unnamed: 0'])
    ratings_df = pd.read_csv(ratings_file)

    # removing missing values

    # considerable number of missing values for age ratings column > drop column
    movies_df.drop(columns = ['ageRating'], inplace = True)
    # small number of missing values for other columns > drop rows
    movies_df.dropna(inplace = True)

    # formatting columns

    # extracting release year from titles
    # defining regex pattern for years
    year_pattern = re.compile('\((\d{4})\)')
    movies_df['releaseYear'] = movies_df['title'].map(lambda x: int(year_pattern.findall(x)[0]) if len(year_pattern.findall(x)) == 1 else np.nan)
    # dropping rows that dont have release years
    movies_df.dropna(inplace = True)
    # converting release year column to integer
    movies_df['releaseYear'] = movies_df['releaseYear'].astype('int')
    # removing release year from titles
    # defining regex pattern for titles
    title_pattern = re.compile('([A-Z]{1}.+)\s\(\d{4}\)')
    movies_df['title'] = movies_df['title'].map(lambda x: title_pattern.findall(x)[0] if len(title_pattern.findall(x)) == 1 else np.nan)
    # converting genre and actor columns to lists
    movies_df['genres'] = movies_df['genres'].map(lambda x: x.split(', '))
    movies_df['actors'] = movies_df['actors'].map(lambda x: x.split(', '))

    # one hot encoding genres

    # re-formatting genres column for vectorisation
    movies_df['genres'] = movies_df['genres'].map(lambda x: ' '.join(x))
    # creating an array of count vectorised movies
    cv = CountVectorizer()
    genres_array = cv.fit_transform(movies_df['genres'])
    # converting array to a dataframe
    genres_df = pd.DataFrame(genres_array.todense(), columns = cv.get_feature_names(), index = movies_df.index)
    # dropping ambiguous genres and genres with counts of less than 1000 
    genres_df.drop(columns = ['adult', 'film', 'fi', 'game', 'news', 'noir', 'reality', 'show', 'talk', 'tv', 'music', 'short'], inplace = True)
    # combining sci-fi columns
    genres_df['sciFi'] = genres_df['sci']
    genres_df.drop(columns = ['sci'], inplace = True)
    # merging one-hot encoded genres with movies
    movies_df = pd.merge(movies_df, genres_df, left_index = True, right_index = True).drop(columns = ['genres'])

    # removing outliers

    # removing movies released before 1980
    before_1980 = movies_df.loc[movies_df['releaseYear'] < 1980].index
    movies_df.drop(labels = before_1980, axis = 0, inplace = True)

    # reducing movies and ratings

    # merging cleaned movies with ratings
    ratings_df = pd.merge(ratings_df, movies_df, left_on = 'movieId', right_on = 'movieId', how = 'right')[['userId', 'movieId', 'rating', 'timestamp']].dropna()

    # subsetting the 10000 most popular movies
    movie_counts = pd.DataFrame(ratings_df['movieId'].value_counts()).reset_index()
    movie_counts.columns = ['movieId', 'reviews']
    top_10000_movies = movie_counts.head(10000)
    movies_df = pd.merge(movies_df, top_10000_movies, left_on = 'movieId', right_on = 'movieId', how = 'right').drop(columns = ['reviews', 'releaseYear']).dropna()
    ratings_df = pd.merge(ratings_df, top_10000_movies, left_on = 'movieId', right_on = 'movieId', how = 'right')[['userId', 'movieId', 'rating', 'timestamp']].dropna()

    # removing users with less than 1000 reviews
    user_counts = pd.DataFrame(ratings_df['userId'].value_counts()).reset_index()
    user_counts.columns = ['userId', 'reviews']
    more_than_1000_users = user_counts.loc[user_counts['reviews'] > 1000]
    ratings_df = pd.merge(ratings_df, more_than_1000_users, left_on = 'userId', right_on = 'userId', how = 'right')[['userId', 'movieId', 'rating', 'timestamp']].dropna()

    # removing movies with less than 25 reviews
    movie_counts = pd.DataFrame(ratings_df['movieId'].value_counts()).reset_index()
    movie_counts.columns = ['movieId', 'reviews']
    more_than_25_movies = movie_counts.loc[movie_counts['reviews'] >= 25]
    ratings_df = pd.merge(ratings_df, more_than_25_movies, left_on = 'movieId', right_on = 'movieId', how = 'right')[['userId', 'movieId', 'rating', 'timestamp']].dropna()

    # saving cleaned and reduced movies and ratings
    movies_df.to_csv(cleaned_movies_file)
    ratings_df.to_csv(cleaned_ratings_file)

    return print("files cleaned")