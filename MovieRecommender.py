import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Download the MovieLens dataset
url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
response = requests.get(url)
with open('movielens.zip', 'wb') as file:
    file.write(response.content)

# Extract the zip file contents
with zipfile.ZipFile('movielens.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Read the ratings.csv file
ratings = pd.read_csv('ml-latest-small/ratings.csv',
                      header=0,
                      sep=',',
                      quotechar='"',
                      usecols=['userId', 'movieId', 'rating'])

ratings.dropna(inplace=True)

# Split data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Create user-item matrix
user_item_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Apply Singular Value Decomposition (SVD)
svd = TruncatedSVD(n_components=50, random_state=42)
matrix_svd = svd.fit_transform(user_item_matrix)

# Test set preprocessing for SVD
test_user_item_matrix = test_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
train_movie_ids = train_data['movieId'].unique()
test_movie_ids = test_data['movieId'].unique()

# Identify movies in test set but not in training set
missing_movie_ids = set(test_movie_ids) - set(train_movie_ids)

# Add missing movie columns to test_user_item_matrix filled with zeros
for movie_id in missing_movie_ids:
    test_user_item_matrix[movie_id] = 0

# Reorder columns of test_user_item_matrix to match training matrix
test_user_item_matrix = test_user_item_matrix.reindex(columns=train_movie_ids).fillna(0)

# Apply the transformation
test_svd = svd.transform(test_user_item_matrix)
predictions = np.dot(test_svd, svd.components_)

mse = mean_squared_error(test_user_item_matrix.values, predictions)
print("SVD Mean Squared Error:", mse)

# SVD-based recommendations
def recommend_movies(user_id, num_recommendations):
    user_row = user_item_matrix.loc[user_id].values.reshape(1, -1)
    user_svd = svd.transform(user_row)
    scores = np.dot(user_svd, svd.components_)
    movie_ids = np.argsort(scores[0])[::-1][:num_recommendations]
    return movie_ids

# Calculate cosine similarity for user-based collaborative filtering
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to recommend movies using user-based collaborative filtering
def user_based_recommendations(user_id, num_recommendations, num_neighbors=10):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_neighbors+1]
    similar_users_ratings = user_item_matrix.loc[similar_users].mean(axis=0)
    user_rated_movies = user_item_matrix.loc[user_id].replace(0, np.nan).dropna().index
    recommendations = similar_users_ratings.drop(user_rated_movies).sort_values(ascending=False).index[:num_recommendations]
    return recommendations

# Calculate cosine similarity for item-based collaborative filtering
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to recommend movies using item-based collaborative filtering
def item_based_recommendations(user_id, num_recommendations, num_neighbors=10):
    user_ratings = user_item_matrix.loc[user_id]
    similar_items = pd.Series(dtype=float)
    for movie, rating in user_ratings[user_ratings > 0].items():
        similar_items = similar_items.add(item_similarity_df[movie].drop(movie) * rating, fill_value=0)
    similar_items = similar_items.groupby(similar_items.index).sum()
    user_rated_movies = user_ratings.replace(0, np.nan).dropna().index
    recommendations = similar_items.drop(user_rated_movies).sort_values(ascending=False).index[:num_recommendations]
    return recommendations

def evaluate_svd_model(svd, num_recommendations=5):
    user_item_matrix_svd = svd.transform(user_item_matrix)
    mse_list = []
    for user_id in test_data['userId'].unique():
        actual_ratings = test_data[test_data['userId'] == user_id]
        user_row = user_item_matrix.loc[user_id].values.reshape(1, -1)
        user_svd = svd.transform(user_row)
        scores = np.dot(user_svd, svd.components_)
        movie_ids = np.argsort(scores[0])[::-1][:num_recommendations]
        predicted_ratings = []
        for movie_id in movie_ids:
            if movie_id in actual_ratings['movieId'].values:
                predicted_ratings.append(actual_ratings[actual_ratings['movieId'] == movie_id]['rating'].values[0])
            else:
                predicted_ratings.append(0)  # Assume rating of 0 for movies not in test set
        mse_list.append(mean_squared_error([0]*len(predicted_ratings), predicted_ratings))
    return np.mean(mse_list)

def evaluate_collab_model(recommendation_func, num_recommendations=5, num_neighbors=10):
    mse_list = []
    for user_id in test_data['userId'].unique():
        actual_ratings = test_data[test_data['userId'] == user_id]
        if recommendation_func.__name__ == 'hybrid_recommendations':
            recommended_movies = recommendation_func(user_id, num_recommendations)
        else:
            recommended_movies = recommendation_func(user_id, num_recommendations, num_neighbors)
        predicted_ratings = []
        for movie_id in recommended_movies:
            if movie_id in actual_ratings['movieId'].values:
                predicted_ratings.append(actual_ratings[actual_ratings['movieId'] == movie_id]['rating'].values[0])
            else:
                predicted_ratings.append(0)  # Assume rating of 0 for movies not in test set
        mse_list.append(mean_squared_error([0]*len(predicted_ratings), predicted_ratings))
    return np.mean(mse_list)

def tune_svd_components():
    components = [20, 50, 100]
    best_mse = float('inf')
    best_components = 0

    for n in components:
        svd = TruncatedSVD(n_components=n, random_state=42)
        svd.fit(user_item_matrix)
        mse = evaluate_svd_model(svd, num_recommendations=5)
        print(f"SVD with {n} components: MSE={mse}")
        
        if mse < best_mse:
            best_mse = mse
            best_components = n

    print(f"Best SVD MSE: {best_mse} with {best_components} components")
    return best_components

best_svd_components = tune_svd_components()

# Tune the number of neighbors
def tune_hyperparameters():
    neighbors = [5, 10]
    best_user_based_mse = float('inf')
    best_item_based_mse = float('inf')
    best_user_neighbors = 0
    best_item_neighbors = 0

    for n in neighbors:
        user_based_mse = evaluate_collab_model(user_based_recommendations, num_neighbors=n)
        item_based_mse = evaluate_collab_model(item_based_recommendations, num_neighbors=n)
        print(f"User-based CF MSE with {n} neighbors:", user_based_mse)
        print(f"Item-based CF MSE with {n} neighbors:", item_based_mse)

        if user_based_mse < best_user_based_mse:
            best_user_based_mse = user_based_mse
            best_user_neighbors = n

        if item_based_mse < best_item_based_mse:
            best_item_based_mse = item_based_mse
            best_item_neighbors = n

    print(f"Best User-based CF MSE: {best_user_based_mse} with {best_user_neighbors} neighbors")
    print(f"Best Item-based CF MSE: {best_item_based_mse} with {best_item_neighbors} neighbors")

    return best_user_neighbors, best_item_neighbors

best_user_neighbors, best_item_neighbors = tune_hyperparameters()

# Hybrid Recommendation System
def hybrid_recommendations(user_id, num_recommendations):
    svd_recommendations = recommend_movies(user_id, num_recommendations * 3)
    user_based_recommendations_list = user_based_recommendations(user_id, num_recommendations * 3, best_user_neighbors)
    item_based_recommendations_list = item_based_recommendations(user_id, num_recommendations * 3, best_item_neighbors)

    combined_recommendations = list(set(svd_recommendations) | set(user_based_recommendations_list) | set(item_based_recommendations_list))
    combined_scores = pd.Series(index=combined_recommendations, dtype=float)

    for movie_id in combined_recommendations:
        svd_score = 1 if movie_id in svd_recommendations else 0
        user_score = 1 if movie_id in user_based_recommendations_list else 0
        item_score = 1 if movie_id in item_based_recommendations_list else 0
        combined_scores[movie_id] = svd_score + user_score + item_score

    return combined_scores.sort_values(ascending=False).index[:num_recommendations]

# Example: Recommend 5 movies for user with ID 1 using hybrid filtering
hybrid_recommended_movies = hybrid_recommendations(1, 5)
print("Hybrid Recommended Movies:", hybrid_recommended_movies)

# Evaluate the hybrid model
hybrid_mse = evaluate_collab_model(hybrid_recommendations, num_recommendations=5)
print("Hybrid Collaborative Filtering MSE:", hybrid_mse)

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_precision_recall(recommendation_func, num_recommendations=5):
    precision_list = []
    recall_list = []
    
    for user_id in test_data['userId'].unique():
        actual_ratings = test_data[test_data['userId'] == user_id]
        if len(actual_ratings) == 0:
            continue
        recommended_movies = recommendation_func(user_id, num_recommendations)
        relevant_items = actual_ratings[actual_ratings['rating'] >= 4]['movieId']
        
        if len(relevant_items) == 0:
            continue
        
        recommended_set = set(recommended_movies)
        relevant_set = set(relevant_items)
        
        true_positives = len(recommended_set & relevant_set)
        false_positives = len(recommended_set - relevant_set)
        false_negatives = len(relevant_set - recommended_set)
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, f1

# Evaluate precision, recall, and F1-score for the hybrid model
precision, recall, f1 = evaluate_precision_recall(hybrid_recommendations, num_recommendations=5)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

from sklearn.model_selection import KFold

def cross_validate_model(recommendation_func, k=5, num_recommendations=5):
    kf = KFold(n_splits=k)
    mse_list = []

    for train_index, test_index in kf.split(ratings):
        train_data = ratings.iloc[train_index]
        test_data = ratings.iloc[test_index]
        
        user_item_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        svd = TruncatedSVD(n_components=best_svd_components, random_state=42)
        matrix_svd = svd.fit_transform(user_item_matrix)
        
        mse = evaluate_collab_model(recommendation_func, num_recommendations=num_recommendations)
        mse_list.append(mse)
    
    avg_mse = np.mean(mse_list)
    print(f"Cross-Validated MSE: {avg_mse}")
    return avg_mse

# Cross-validate the hybrid model
cross_validate_model(hybrid_recommendations, k=5, num_recommendations=5)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot heatmap of user similarities
plt.figure(figsize=(10, 8))
sns.heatmap(user_similarity_df, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title("User Similarities")
plt.show()

# Plot heatmap of item similarities
plt.figure(figsize=(10, 8))
sns.heatmap(item_similarity_df, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title("Item Similarities")
plt.show()

# Plot distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], bins=20, kde=True)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    num_recommendations = int(request.args.get('num_recommendations', 5))
    recommendations = hybrid_recommendations(user_id, num_recommendations)
    return jsonify(recommendations.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

