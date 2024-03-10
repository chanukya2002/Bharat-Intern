# Install Surprise library
# !pip install scikit-surprise

import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Load the MovieLens dataset (you can download it from https://grouplens.org/datasets/movielens/)
data = pd.read_csv('path_to_movie_lens_dataset/ml-latest-small/ratings.csv')

# Define the Reader object
reader = Reader(rating_scale=(0.5, 5))

# Load the dataset into Surprise's Dataset format
surprise_data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Use SVD algorithm
algo = SVD()

# Perform cross validation
cross_validate(algo, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model
trainset = surprise_data.build_full_trainset()
algo.fit(trainset)

# Make recommendations for user 1
user_id = 1
user_ratings = data[data['userId'] == user_id]
movies_not_rated = data[~data['movieId'].isin(user_ratings['movieId'])]['movieId'].unique()

predictions = []
for movie_id in movies_not_rated:
    prediction = algo.predict(user_id, movie_id)
    predictions.append((prediction.iid, prediction.est))

# Sort the predictions by estimated rating
predictions.sort(key=lambda x: x[1], reverse=True)

# Display top N recommended movies
top_n = 10
top_movies = predictions[:top_n]

print(f"Top {top_n} Recommendations for User {user_id}:")
for movie_id, estimated_rating in top_movies:
    movie_title = data[data['movieId'] == movie_id]['title'].iloc[0]
    print(f"{movie_title}: Estimated Rating {estimated_rating:.2f}")
