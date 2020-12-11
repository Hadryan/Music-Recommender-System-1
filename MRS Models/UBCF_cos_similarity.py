# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:38:25 2020

@author: SKD-HiTMAN
"""

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv('../../../Dataset/SongsDataset/ratings.csv', encoding='latin-1')
songs = pd.read_csv('../../../Dataset/NewDataset/songs.csv', encoding='latin-1')
# ratings = pd.read_csv('../Dataset/MovieLens/ml-latest-small/ratings.csv')
# songs = pd.read_csv('../Dataset/MovieLens/ml-latest-small/songs.csv', encoding='latin-1')

merged = pd.merge(ratings, songs, left_on='songId', right_on='songId', sort=True)
merged = merged[['userId', 'title', 'rating']]
songRatings = merged.pivot_table(index=['userId'], columns=['title'], values='rating')

# remove null value-> replace with 0
#songRatings.replace({np.nan:0}, regex=True, inplace=True)
songRatings = songRatings.fillna(0)

#print((songRatings[songRatings['userId'] == None]).head())

#print(songRatings.head())

# cosine similarity: pairwise similarity b/w all users and song-rating dfs
user_similarity = cosine_similarity(songRatings)

# user_similarity is a numpy array--> convert to df
user_sim_df = pd.DataFrame(user_similarity, index = songRatings.index, columns = songRatings.index)

songRatings = songRatings.T   # transpose the df to work on columns in the upcoming function


# function to take user as parameter and show the result of highest rated songs for similar user
import operator
def recommendation(user):
    if user not in songRatings.columns:
        return ('Oops! No data available for this user!')
    
    # sort all the similar user for the active user basing on cosine similarity
    sim_user = user_sim_df.sort_values(by = user, ascending = False).index[1:11]
    
    best = []
    for i in sim_user:
        max_score = songRatings.loc[:, i].max()
        best.append(songRatings[songRatings.loc[:, i] == max_score].index.tolist())
        
    user_seen_songs = songRatings[songRatings.loc[:, user] > 0].index.tolist()
    
    # remove the songs user has already watched
    for i in range(len(best)):
        for j in best[i]:
            if (j in user_seen_songs):
                best[i].remove(j)
    most_common = {}
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1
    
    sorted_list = sorted(most_common.items(), key = operator.itemgetter(1), reverse = True)    # sort by 1st elemt which is similar to user--> op.itemgetter(1)
    return(sorted_list)

result = recommendation(45)


################################## performance evaluation

# create classes
# 1-3 ratings -> disliked; 4-5 -> liked
ratings.loc[ratings['rating'] <= 3, "rating"] = 0
ratings.loc[ratings['rating'] > 3, "rating"] = 1


merged = pd.merge(ratings, songs, left_on='songId', right_on='songId', sort=True)
merged = merged[['userId', 'title', 'genres', 'rating']]

# separating genres column
merged = pd.concat([merged, merged['genres'].str.get_dummies(sep='|')], axis = 1)

# we still have genres and no genres listed: remove them
del merged['genres']
# del merged['(no genres listed)']

# keeping classes column in the last
cols = list(merged.columns.values)
cols.pop(cols.index('rating'))
merged = merged[cols+['rating']]

################ Building model

# instantiate the matrices of features(independent and dependent variables)
X = merged.iloc[:, 2:61].values   # independent var: genres
Y = merged.iloc[:, 61].values     # dependent var: rating


################## splitting train and test data
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


merged_p = merged[merged['userId'] == 45]
m = merged_p.iloc[14][1]
p = result[9][0]

d = merged.iloc[1][1]

r_pred = []
i = 0
while(i < 10):
    j = 0
    while (j < 15):
        if result[i][0] == merged_p[j][1]:
            r_pred.append(merged_p[j][61])
            break
        j += 1
    i += 1


















