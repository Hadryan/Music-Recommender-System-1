# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:19:58 2020

@author: SKD-HiTMAN
"""
import pandas as pd
import numpy as np

from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

ratings_df = pd.read_csv('../../Dataset/SongsData/ratings.csv')
songs_df = pd.read_csv('../../Dataset/SongsData/songs.csv', encoding='latin-1')


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
# now we have ratings by each user to each

pred_df_T = np.transpose(predicted_rating_df)


# cosine similarity: pairwise similarity b/w all users and song-rating dfs
item_similarity = cosine_similarity(pred_df_T)

# user_similarity is a numpy array--> convert to df : to be able to use pandas useful features
item_sim_df = pd.DataFrame(item_similarity, index = pred_df_T.index, columns = pred_df_T.index)


# looking for the song by song Id
def sim_songs_to(songId):
    count = 1
    songIndex = songs_df.index[songs_df['songId'] == songId]
    print('Similar songs to {} are: '.format(songs_df.loc[songIndex].title))
    
    # sorts by song title and get top ten results
    for item in item_sim_df.sort_values(by = songId, ascending=False).index[1:11]:
        itemIndex = songs_df.index[songs_df['songId'] == item]
        print('No. {} : {}'.format(count, songs_df.loc[itemIndex].title))
        count += 1

sim_songs_to(4)





































    
    
    
    
    
    
    
    
    
    
    
    
    