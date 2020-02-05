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

The modelling process for this project involved building three recommendation systems: a content-based system, a user-based system (i.e. matrix factorisation) and a hybrid system (i.e. nueral network). 

In the absence of any online feedback from users, the average cosine dissimilarity between recommendations was used to assess the effectiveness of each system. Note that this was used instead of the conventional rmse metric to account for the level of personalisation in the recommendations provided for each user.

### Content-based System

The general pipeline:

1. **Tokenisation** - converting plotlines into a list of lowercase word tokens. <br/><br/>
<img src="/images/unprocessed_plot_example.png" /> <br/><br/>
2. **Dimensionality reduction** - removing stopwords and lemmatisation. <br/><br/>
<img src="/images/processed_plot_example.png" /> <br/><br/>
3. **Feature engineering** - count vectorising genres, actors, directors and normalising plotlines. <br/><br/>
<img src="/images/one-hot_encoding_example.png" /> <br/><br/>
4. **Similarity scores** - computing the similarity scores between each pair of movies using the cosine similarity metric. <br/><br/>
<img src="/images/similarity_scores_example.png" /> <br/><br/>
5. **Recommendations** - sorting movies in order of similarity and recommending the top k in the list. <br/><br/>
<img src="/images/content_recommendations.png" />

### User-based System

### Hybrid System

## Conclusions and Possible Extensions
