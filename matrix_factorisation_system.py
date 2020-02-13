from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

# creating an rdd object for parallel processing of ratings
def create_rdd(ratings_df):
  # converting userId column from float to integer
  ratings_df['userId'] = ratings_df['userId'].astype('int')
  # creating a list of (user, movie, rating) tuples
  user_movie_rating_tuples = []
  for index, row in ratings_df.iterrows():
    user_movie_rating_tuples.append(Rating(row['userId'], row['movieId'], row['rating']))
  # converting list of tuples into a rdd
  rdd = sc.parallelize(user_movie_rating_tuples)
  return rdd

# training matrix factorisation model, using PySpark API
def train_mf_model(train_set, test_set, n_embeddings = 5, n_iterations = 5, reg_param = 0.01):
  model = ALS.train(train_set, n_embeddings, iterations = n_iterations, lambda_ = reg_param,  seed = 1)
  test_predictions = model.predictAll(test_set.map(lambda x: (x.user, x.product))).map(lambda x: ((x.user, x.product), x.rating))
  test_ratings_and_predictions = test_set.map(lambda x: ((x.user, x.product), x.rating)).join(test_predictions)
  rmse = np.sqrt(test_ratings_and_predictions.map(lambda x: (x[1][0] - x[1][1])**2).mean())
  return model, rmse

# outputting a list of recommended movie indices for all movies
def top_n_movies_list_mf(model, users, n_rec):
  top_n_movies_list = []
  for user in users:
    top_n_user_movie_ids = [movie[1] for movie in model.recommendProducts(int(user), n_rec)]
    top_n_user_movie_tokens = ['movie' + str(movie_id) for movie_id in top_n_user_movie_ids]
    top_n_movies_list.append(top_n_user_movie_tokens)
  return top_n_movies_list