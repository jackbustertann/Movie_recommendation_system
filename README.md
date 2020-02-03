# Building a Movie Recommendation System

## The Motivation

The motivation for this project was to compare the effectiveness of some of the traditional approaches used in e-commerce for making personalised recommendations to online users. I was also keen to develop more of an understanding of how these systems are implemented at scale by getting some hands on experience with a distributed computing framework, such as Apache Spark.

## Data Collection

The dataset used for this project consisted of 25,000,000 reviews on over 60,000 movies featured on the IMDb website. This included the ratings provided by over 100,000 users along with the titles and category tags for each movie. The repository used to source this data can be found [here](https://grouplens.org/datasets/movielens/). <br/> <br/>
I also collected the directors, actors and plotlines for each movie by making a series of requests to the OMDb database using their [public API](http://www.omdbapi.com/).

## Data Cleaning

## EDA

Upon exploring the data, I discovered that:

- Users were generally more inclined to give positive reviews. <br/><br/>
<img src="/images/rating_counts.png" width="600"/>

- The most popular movies were generally rated amongst the top 5% of all movies. <br/><br/>
<img src="/images/most_popular_movies.png" width="800"/>

- The most active users were generally more critical than the average user. <br/><br/>
<img src="/images/most_active_users.png" width="600"/>

## Modelling

### Content-based System

### User-based System - Matrix Factorisation

### Hybrid System - Nueral Network

## Conclusions and Possible Extensions
