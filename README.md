# Building a Movie Recommendation System

## The Motivation

The motivation for this project was to compare the effectiveness of some of the traditional approaches used in e-commerce for making personalised recommendations to online users. I was also keen to develop more of an understanding of how these systems are implemented at scale by getting some hands on experience with a distributed computing framework, such as Apache Spark.

## Data Collection

The dataset used for this project consisted of 25,000,000 reviews on over 60,000 movies featured on the IMDb website. This included the ratings provided by over 100,000 users along with the titles and category tags for each movie. The repository used to source this data can be found [here](https://grouplens.org/datasets/movielens/). <br/> <br/>
I also collected the directors, actors and plotlines for each movie by making a series of requests to the OMDb database using their [public API](http://www.omdbapi.com/).

## Data Cleaning

The cleaning process for this project involved:

- One-hot encoding movie genres. <br/><br/>
- Dropping movies released before 1980. <br/><br/>
- Dropping users with less than 1,000 reviews. <br/><br/>
- Dropping movies with less than 25 reviews.

## EDA

Upon exploring the data, I discovered that:

1. Users were generally more inclined to give positive reviews. <br/><br/>
<img src="/images/rating_counts.png" width="600"/>

2. The most popular movies were generally rated amongst the top 5% of all movies. <br/><br/>
<img src="/images/most_popular_movies.png" width="800"/>

3. The most active users were generally more critical than the average user. <br/><br/>
<img src="/images/most_active_users.png" width="600"/>

## Modelling

The modelling process for this project involved building three recommendation systems: a content-based system and two user-based systems. 

It is important to note that whilst the user-based systems were validated using the conventional rmse metric, the content-based system did not explicitly use the ratings provided by users and so catalog coverage and average cosine dissimilarity between recommendations were used instead for this case. These metrics were chosen for the content-based system to reflect the level of personalisation in the recommendations and to penalise the system if it had any bias towards a small subset of movies.

### Content-based System

High level definition: <br/><br/>
*Recommending movies to users based on explicit features of movies they have watched or rated highly in the past.* <br/><br/>

General pipeline:

1. **Tokenisation** - converting plotlines into a list of lowercase word tokens. <br/><br/>
<img src="/images/unprocessed_plot_example.png" /> <br/><br/>
2. **Dimensionality reduction** - removing stopwords and lemmatisation. <br/><br/>
<img src="/images/processed_plot_example.png" /> <br/><br/>
3. **Feature engineering** - count vectorising genres, actors, directors, and plotlines (with normalisation). <br/><br/>
<img src="/images/one-hot_encoding_example.png" /> <br/><br/>
4. **Similarity scores** - computing the similarity scores between each pair of movies using the cosine similarity metric. <br/><br/>
<img src="/images/similarity_scores_example.png" /> <br/><br/>
5. **Recommendations** - sorting movies in order of similarity and recommending the top k in the list. <br/><br/>
<img src="/images/content_recommendations.png" /> <br/><br/>

### User-based Systems

High level definition: <br/><br/>
*Recommending movies to users based on the actions of other users with similar preferences, either provided explicity during registration to the app or website or inferred implicity from their online behaviour.* <br/><br/>

#### Matrix Factorisation Model

Model assumptions:

- *User ratings can be decomposed into a dot product between user preferences and movie features.* <br/> <br/>
- *Users who give similar ratings for different movies must have similar preferences.* <br/> <br/>
- *Movies that are given similar ratings by different users must have similar features.* <br/> <br/>

General pipeline:

1. **Spark context** - creating a connection to a Spark cluster. <br/><br/>
2. **RDD** - partitioning ratings across nodes of cluster for parallel processing. <br/><br/>
3. **Hyperparameter tuning** - finding the optimal number of embeddings to learn for each user and movie. <br/><br/>
4. **Model training** - updating embeddings until they converge. <br/><br/>
5. **Recommendations** - sorting movies in order of predicted rating and recommending the top k in the list. <br/><br/>

Visualising movie embeddings using PCA:

<img src="/images/mf_movie_embeddings.png" /> <br/><br/>

#### Neural Network Model

Model assumptions: <br/> <br/>
*This model can be seen as an extension to the matrix factorisation model that aims to extract extra levels of abstraction from the user and movie embeddings by passing the element-wise dot product of these embeddings through additional hidden layers.* <br/> <br/>

Model architecture:

1. **Input layer** - unique integer ids for users and movies. <br/><br/>
2. **Embeddings** - latent features for users and movies inferred from user ratings. <br/><br/>
3. **Dot product** - the element-wise dot product between embeddings. <br/><br/>
4. **Hidden layer(s)** - abstract representation(s) of inputs derived from embeddings. <br/><br/>
5. **Output layer** - predicted rating provided by user for movie. <br/><br/>
<img src="/images/nn_architecture.png" /> <br/><br/>

## Conclusions and Possible Extensions

The content-based system generally outperformed the used-based systems for most instances of users. This is likely to be a result the large amount of sparsity in the user-movie feedback matrix. To address this limitation in the dataset, implicit data such as watch and purchase history could be used in addition to the user ratings to develop a more detailed understanding of each movie. 

Possible extensions:

- **Deeper nueral networks** <br/><br/>
- **Hybrid systems** <br/><br/>
- **Online metrics**

