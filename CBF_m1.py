# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:42:45 2020

@author: SKD-HiTMAN
"""

# CBF focusses mainly upon item features
# Method 01: Using ML, classify the dataset and predict items based on the items user earlier was interested in

import pandas as pd
import numpy as np


# ratings = pd.read_csv('../../../Dataset/MovieLens/ml-latest-small/ratings.csv')
# songs = pd.read_csv('../../../Dataset/MovieLens/ml-latest-small/songs.csv', encoding='latin-1')

ratings = pd.read_csv('../../../Dataset/SongsDataset/ratings.csv', encoding='latin-1')
songs = pd.read_csv('../../../Dataset/NewDataset/songs.csv', encoding='latin-1')

songs.info()


"""
flag = 0
count = 0
for i in ratings['userId']:
    count += 1
    for j in songs['userId']:
        if i == j:
            flag = 1
            break
        
if flag == 0:
    p = "PROBLEM!"
else:
    p = "No problem!"
c = count

"""

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


############ Normalize the data to a common scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Using a flexible and efficient classification model- Random Forest
# Build Binary classification model, since we have only 2 classes
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion='entropy', random_state = 0)

################### fit the model to the training set
classifier.fit(X_train, Y_train)

################# predict the test set results
Y_pred = classifier.predict(X_test)

############################# check accuracy of predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)    # allows the visualization of performance of the algorithm
acc = accuracy_score(Y_test, Y_pred)  # =>> 61.66%

totalSongIds = songs['songId'].unique()


"""
userId = 7

ratedsongs = ratings['songId'].loc[ratings['userId'] == userId]
non_ratedsongs = np.setdiff1d(totalSongIds, ratedsongs.values)
non_ratedsongsDF = pd.DataFrame(non_ratedsongs, columns=['songId'])
non_ratedsongsDF['userId'] = userId
non_ratedsongsDF['prediction'] = 0
active_user_nonratedsongs = non_ratedsongsDF.merge(songs, left_on='songId', right_on='songId', sort=True)
active_user_nonratedsongs = pd.concat([active_user_nonratedsongs, active_user_nonratedsongs['genres'].str.get_dummies(sep = '|')], axis = 1)

del active_user_nonratedsongs['genres']
# del active_user_nonratedsongs['(no genres listed)']
del active_user_nonratedsongs['title']
        
active_user_nonratedsongsDF = active_user_nonratedsongs
"""


def nonratedsongs(userId):
    
    ratedsongs = ratings['songId'].loc[ratings['userId'] == userId]
    non_ratedsongs = np.setdiff1d(totalSongIds, ratedsongs.values)
    non_ratedsongsDF = pd.DataFrame(non_ratedsongs, columns=['songId'])
    non_ratedsongsDF['userId'] = userId
    non_ratedsongsDF['prediction'] = 0
    active_user_nonratedsongs = non_ratedsongsDF.merge(songs, left_on='songId', right_on='songId', sort=True)
    active_user_nonratedsongs = pd.concat([active_user_nonratedsongs, active_user_nonratedsongs['genres'].str.get_dummies(sep = '|')], axis = 1)

    del active_user_nonratedsongs['genres']
    # del active_user_nonratedsongs['(no genres listed)']
    del active_user_nonratedsongs['title']
    
    return (active_user_nonratedsongs)

active_user_nonratedsongsDF = nonratedsongs(15)

df = active_user_nonratedsongsDF.iloc[:, 5:].values
Y_pred2 = classifier.predict(df)

active_user_nonratedsongsDF['prediction'] = Y_pred2
recommend = active_user_nonratedsongsDF[['songId', 'prediction']]
recommend = recommend.loc[recommend['prediction'] == 1]
# recommend = recommendation.loc[recommendation['prediction'] == 1]

recommend.to_csv('../../../Dataset/Created/recommend.csv', sep = ',', index = False)

# now we can sort this data in any way to show the results to the user.




















