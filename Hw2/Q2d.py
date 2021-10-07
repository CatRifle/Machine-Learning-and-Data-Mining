import os
from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
import sklearn
import math
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn import datasets,linear_model,preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

"""Hello, tutor. I used google colab to train and test, I imported the Q2.csv file from google drive
    if you want to test the .csv on you pc, you probably need to 
   change the file path and delete the command like drive.mount above, thanks
"""
f_array= np.genfromtxt('/content/drive/MyDrive/9417 HW2/Q2.csv',delimiter=",")
f_array=f_array[1:,:]

''' Delete rows with Nan'''
f_new = []
for row in f_array:
  count = 0
  for item in row:
    if np.isnan(item):
      count = count+1
  if count==0:
    f_new.append(row)


'''Processing dataset after removing rows with nan'''
f_new= np.array(f_new)
Set_X = f_new[:,1:4]
Set_Y = f_new[:,6]

sc = MinMaxScaler(feature_range=(0,1))
Set_X = sc.fit_transform(Set_X)
X_train, X_test = Set_X[:204], Set_X[204:]
Y_train, Y_test = Set_Y[:204], Set_Y[204:]