# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:36:14 2020

@author: SKD-HiTMAN
"""

import pandas as pd
import numpy as np

# ratings = pd.read_csv('../../../Dataset/ratings.csv')
# songs = pd.read_csv('../../../Dataset/songs.csv', encoding='latin-1')

# ratings = pd.read_csv('../../../Dataset/SongsData/ratings.csv')
# songs = pd.read_csv('../../../Dataset/SongsData/songs.csv', encoding='latin-1')

ratings = pd.read_csv('../../../Dataset/SongsDataset/ratings.csv', encoding='latin-1')
songs = pd.read_csv('../../../Dataset/NewDataset/songs.csv', encoding='latin-1')

ts = ratings['timestamp']

ts = pd.to_datetime(ts, unit='s').dt.hour
songs['hours'] = ts

merged = ratings.merge(songs, left_on='songId', right_on='songId', suffixes=['_users', ''])
merged = merged[['userId', 'songId', 'genres', 'hours']]

# separating genres column
merged = pd.concat([merged, merged['genres'].str.get_dummies(sep='|')], axis = 1)
del merged['genres']
# del merged['(no genres listed)']


"""
userId = 15
userprofile = merged.loc[merged['userId'] == userId]   # extract user rating info
del userprofile['userId']

#get the preference of the user for hours of the day
userprofile = userprofile.groupby(['hours'], as_index = False, sort = True).sum()

#normalizing the userprofile dataframe
userprofile.iloc[:, 2:60] = userprofile.iloc[:, 2:60].apply(lambda x:(x - np.min(x))/(np.max(x) - np.min(x)), axis = 1)

activeuser = userprofile
"""

# Build context profile of the user
def activeuserprofile(userId):
    userprofile = merged.loc[merged['userId'] == userId]   # extract user rating info
    del userprofile['userId']
    #get the preference of the user for hours of the day
    userprofile = userprofile.groupby(['hours'], as_index = False, sort = True).sum()
    #normalizing the userprofile dataframe
    userprofile.iloc[:, 2:] = userprofile.iloc[:, 2:].apply(lambda x:(x - np.min(x))/(np.max(x) - np.min(x)), axis = 1)
    
    return (userprofile)

activeuser = activeuserprofile(31)

##################  importing recommended songs list from CBR model   ####################

recommend = pd.read_csv('../../../Dataset/Created/recommend.csv', sep = ',')
merged = merged.drop_duplicates()
# del merged['userId']

# user_pref = recommend.merge(merged, left_on = 'songId', right_on = 'songId', suffixes = ['_user', ''])
user_pref = recommend.merge(merged, left_on='songId', right_on='songId', suffixes=['_users', ''])
activeuser_ = np.transpose(activeuser)

"""
activeuser_ = np.transpose(activeuser)
print(activeuser_.shape)

up = user_pref.iloc[:, 5:]
au = activeuser_.iloc[2:, 21]
print(up.shape, au.shape)

"""

product = np.dot(user_pref.iloc[:, 4:].to_numpy(), activeuser_.iloc[2:, 1].to_numpy())
preferences = np.stack((user_pref['songId'], product), axis = 1)

df = pd.DataFrame(preferences, columns = ['songId', 'preferences'])
res = df.drop_duplicates()
res = res[res['preferences'] == 1]
result = (res.sort_values(['preferences'], ascending = False)).iloc[0:10, 0]


































