# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:59:26 2020

@author: SKD-HiTMAN
"""

# CBF focusses mainly upon item features
# Method 02: Using ML, classify the dataset and predict items based on the items user earlier was interested in

import pandas as pd
import numpy as np

songs = pd.read_csv('../../../Dataset/NewDataset/songs.csv', encoding='latin-1')

songs = pd.concat([songs, songs['genres'].str.get_dummies(sep='|')], axis = 1)
del songs['genres']


selected = []


# users preferences
for i in range(3):
    selected.append(0)
selected.append(1)
selected.append(1)
for i in range(59):
    selected.append(0)
X = songs.iloc[:, 2:]

# import NN class
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(X)

# returns array of songs
songs_arr = nbrs.kneighbors([selected])
print(nbrs.kneighbors([selected]))






























