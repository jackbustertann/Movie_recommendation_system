import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools

def personalisation_score(top_n_movies_list):
    top_n_movies_tokens = [' '.join(movie) for movie in top_n_movies_list]
    vectoriser = CountVectorizer()
    top_n_movies_matrix = vectoriser.fit_transform(top_n_movies_tokens)
    disimilarity_matrix = 1 - cosine_similarity(top_n_movies_matrix, top_n_movies_matrix)
    disimilarity_matrix_sum = np.triu(disimilarity_matrix).sum()
    n_scores = (disimilarity_matrix.shape[0]**2 - disimilarity_matrix.shape[0])/2
    average_disimilarity = disimilarity_matrix_sum / n_scores
    return average_disimilarity

def coverage_score(top_n_movies_list, movies):
    n_movies = len(list(movies))
    flattened_movies = list(itertools.chain.from_iterable(top_n_movies_list))
    unique_movies = list(np.unique(flattened_movies))
    n_recommended_movies = len(unique_movies)
    coverage_score = n_recommended_movies / n_movies
    return coverage_score



