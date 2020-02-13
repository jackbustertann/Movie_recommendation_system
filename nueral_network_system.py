# import libaries
from keras import models, layers, optimizers, initializers, Model

# creating nueral network with a custom architecture
def create_nn(max_user_id, max_movie_id, n_embeddings = 5, n_hidden_layers = 1, n_hidden_units = 5):
    # input layers for users and movies
    users = layers.Input(shape = (1,))
    movies = layers.Input(shape = (1,))
    # embedding layers for users and movies
    embeddings_init = initializers.RandomUniform(minval = -1, maxval = 1, seed = 1)
    user_embeddings = layers.Embedding(max_user_id + 1, n_embeddings, input_length = 1, embeddings_initializer = embeddings_init)(users)
    movie_embeddings = layers.Embedding(max_movie_id + 1, n_embeddings, input_length = 1, embeddings_initializer = embeddings_init)(movies)
    # element-wise dot product between embeddings
    dot_product = layers.multiply([user_embeddings, movie_embeddings])
    # hidden layer, with relu activation
    hidden_layer = layers.Dense(n_hidden_units, activation = "relu")(dot_product)
    # extra hidden layers, with relu activation
    if n_hidden_layers > 1:
        n_layers_left = n_hidden_layers - 1
        while n_layers_left > 0:
            hidden_layer = layers.Dense(n_hidden_units, activation = "relu")(hidden_layer)
            n_layers_left -= 1
    # output layer with sigmoid activation
    output = layers.Dense(1, activation = "sigmoid")(hidden_layer)
    # defining network architecture
    network = Model(inputs = [users, movies], outputs = [output])
    # compiling network
    network.compile('rmsprop','mse')
    return network

# training and validating nueral network
def train_nn(network, train_users, train_movies, train_ratings, test_users, test_movies, test_ratings, batch_size = 256, n_epochs = 5):
    # training compiled network, with a pre-defined batch size and number of epochs
    history = network.fit([train_users, train_movies], train_ratings, validation_data = ([test_users, test_movies], test_ratings), batch_size = batch_size, epochs = n_epochs, verbose = 1)
    return history

# outputting a list of recommended movie indices for a given user
def top_n_movies_nn(network, user, movies, n_rec):
    user_input = user.reshape(1,1)
    movies_input = movies.reshape(-1,1)
    predicted_user_ratings = []
    for movie_input in movies_input:
        predicted_user_ratings.append(network.predict([user_input, movie_input])[0][0][0])
    predicted_user_ratings = list(zip(movies, predicted_user_ratings))
    sorted_user_ratings = sorted(predicted_user_ratings, key = lambda rating: rating[1], reverse = True)[:n_rec]
    sorted_movie_indices = [rating[0] for rating in sorted_user_ratings]
    return sorted_movie_indices