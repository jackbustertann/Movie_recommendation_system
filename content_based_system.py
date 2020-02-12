import numpy as np
import pandas as pd

from data_cleaning import multi_label_one_hot_encoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from string import punctuation
import spacy
import re

# preprocessing movies using one-hot encoded genres, actors and directors
def content_preprocessor(unprocessed_movie_file, processed_movie_file):
    # importing movies
    movies_df = pd.read_csv(unprocessed_movie_file).drop(columns = ['Unnamed: 0'])
    # one-hot encoding actors and directors with a mimimum frequency of 10
    one_hot_actors_df = multi_label_one_hot_encoder(movies_df['actors'], 10)
    one_hot_directors_df = multi_label_one_hot_encoder(movies_df['director'], 10)
    # merging one-hot encoding actors and directors with movies and dropping redundant columns
    preprocessed_content_df = pd.concat([movies_df, one_hot_actors_df, one_hot_directors_df], axis = 1).drop(columns = ['movieId', 'actors', 'director', 'plot'])
    # saving preprocessed movies to csv
    preprocessed_content_df.to_csv(processed_movie_file)
    return print('file processed')

# preprocessing movie plots
def text_preprocessor(text):
    # convert each word to lowercase token
    text_tokens_lower = [word.lower() for word in text.split(' ')]
    # lemmatise tokens
    sp = spacy.load('en')
    text_tokens_lemma = [word.lemma_ for word in sp(' '.join(text_tokens_lower))]
    # remove stop words and punctuation
    stop_words = stopwords.words('english')
    stop_words += punctuation
    text_tokens_stop = [word for word in text_tokens_lemma if word not in stop_words]
    # remove tokens that do not contain three or more letters
    text_tokens_regex = re.compile('[a-z]{3,}').findall(' '.join(text_tokens_stop))
    # convert tokens to string for vectoriser
    preprocessed_text = ' '.join(text_tokens_regex)
    return preprocessed_text

# one-hot encoding preprocessed movie plots
def create_tf_matrix(preprocessed_column, min_freq = 2, max_freq = 0.1):
    vectorizer = CountVectorizer(min_df = min_freq, max_df = max_freq)
    tf_matrix = vectorizer.fit_transform(preprocessed_column)
    features = vectorizer.get_feature_names()
    return tf_matrix, features

# outputing a list of recommended movie indices for a given movie
def top_n_movies_content(tf_matrix, movie_index, n_rec):
    movie_scores = list(enumerate(cosine_similarity(tf_matrix[movie_index,].reshape(1,-1), tf_matrix)[0]))
    sorted_movie_scores = sorted(movie_scores, key = lambda x: x[1], reverse = True)[1:n_rec+1]
    sorted_movie_indices = [movie[0] for movie in sorted_movie_scores]
    return sorted_movie_indices

# outputting a list of recommended movie indices for all movies
def top_n_movies_list_content(tf_matrix, n_rec):
    top_n_movies_list = []
    n_movies = tf_matrix.shape[0]
    for i in range(n_movies):  
        sorted_movie_indices = top_n_movies_content(tf_matrix, i, n_rec)
        sorted_movie_tokens = ['movie' + str(movie) for movie in sorted_movie_indices]
        top_n_movies_list.append(sorted_movie_tokens)
    return top_n_movies_list

# outputing a list of recommended movie titles for a given movie
def content_movie_recommender(content_df, tf_matrix, movie_index, n_rec):
    movie_title = content_df.loc[movie_index, 'title']
    top_n_movie_indices = top_n_movies_content(tf_matrix, movie_index, n_rec)
    print('Users who like:')
    print(movie_title)
    print ('')
    print('Will also like:')
    for index in top_n_movie_indices:
        print(content_df.loc[index, 'title'])
    return print('')