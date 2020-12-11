# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:28:42 2020

@author: SKD-HiTMAN
"""
#Using SVD

import pandas as pd
import numpy as np

from scipy.sparse.linalg import svds


ratings_df = pd.read_csv('../../../Dataset/SongsDataset/ratings.csv', encoding='latin-1')
songs_df = pd.read_csv('../../../Dataset/NewDataset/songs.csv', encoding='latin-1')
# ratings_df = pd.read_csv('../../Dataset/ratings.csv')
# songs_df = pd.read_csv('../../Dataset/songs.csv', encoding='latin-1')


A_df = ratings_df.pivot_table(index='userId', columns='songId', values='rating', aggfunc=np.max).fillna(0)
#A_df.replace({np.nan:0}, regex = True, inplace=True)


A = A_df.to_numpy()   # transform to numpy array to perform matrix opn
user_rating_mean = np.mean(A, axis=1)    # fill in all empty cells by avg rating of users
A_normalized = A - user_rating_mean.reshape(-1, 1)   # subtracting mean from our original data

# performing SVD
U, sigma, Vt = svds(A_normalized, k = 5)
sigma = np.diag(sigma)

predicted_rating = np.dot(np.dot(U, sigma), Vt) + user_rating_mean.reshape(-1, 1)

# dataframe of our predicted rating matrix
predicted_rating_df = pd.DataFrame(predicted_rating, columns = A_df.columns)

# define function to predict best songs for any user which he/she has not already rated/watched
def recommend_songs(predicted_rating_df, userID, songs_df, original_ratings_df, num_recommendations = 5):
    
    user_row_number = userID - 1
    sorted_user_predictions = predicted_rating_df.iloc[user_row_number].sort_values(ascending = False)
    
    # merge user data to songs info like title and genres
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]     # gets the movies rated by the user
    user_full = (user_data.merge(songs_df, how = 'left', left_on = 'songId', right_on = 'songId').
                 sort_values(['rating'], ascending = False))
    print('user {0} has already rated {1} songs.'.format(userID, user_full.shape[0]))    # user_full --> no. of movies rated  by the user
    
    print('Recommending highest {0} predicted ratings songs not already rated.'.format(num_recommendations))
    
    recommendations = (songs_df[~songs_df['songId'].isin(user_full['songId'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',    # to exclude movies user has already seen
                             left_on = 'songId', right_on = 'songId').
                             rename(columns = {user_row_number: 'Predictions'}).
                             sort_values('Predictions', ascending = False).
                             iloc[:num_recommendations, :-1])
    return user_full, recommendations    #user_full => what user already rated; recommendations => what system recommends

already_rated, predictions = recommend_songs(predicted_rating_df, 5, songs_df, ratings_df, 10)
already_rated = already_rated.head(10)
predictions = predictions



































