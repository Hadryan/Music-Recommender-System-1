# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:00:08 2020

@author: SKD-HiTMAN
"""
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv('../Dataset/MovieLens/ml-latest-small/ratings.csv')
songs = pd.read_csv('../Dataset/MovieLens/ml-latest-small/songs.csv', encoding='latin-1')

merged = pd.merge(ratings, songs, left_on='songId', right_on='songId', sort=True)
merged = merged[['userId', 'title', 'rating']]
songRatings = merged.pivot_table(index=['title'], columns=['userId'], values='rating')

# remove null value-> replace with 0
#songRatings.replace({np.nan:0}, regex=True, inplace=True)
songRatings = songRatings.fillna(0)


# cosine similarity: pairwise similarity b/w all users and song-rating dfs
item_similarity = cosine_similarity(songRatings)

# user_similarity is a numpy array--> convert to df : to be able to use pandas useful features
item_sim_df = pd.DataFrame(item_similarity, index = songRatings.index, columns = songRatings.index)


def sim_songs_to(title):
    count = 1
    print('Similar songs to {} are: '.format(title))
    
    # sorts by song title and get top ten results
    for item in item_sim_df.sort_values(by = title, ascending=False).index[1:11]:
        print('No. {} : {}'.format(count, item))
        count += 1

sim_songs_to('22 Jump Street (2014)')

sim_songs_to('Big Hero 6')





































