import os


from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn import datasets,linear_model,preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


from google.colab import drive
drive.mount('/content/drive')
file=pd.read_csv('/content/drive/MyDrive/9417 HW2/Q1.csv')

"""Hello, Omar. I used google colab to train and test, I imported the Q1.csv file from google drive
    if you want to test the .csv on you pc, you probably need to 
   change the file path and delete the command like drive.mount above, thanks
"""

'''Read file from onedrive'''
file_array= np.genfromtxt('/content/drive/MyDrive/9417 HW2/Q1.csv',delimiter=",")
file_arr = file_array[1:]
file_train, file_test = file_arr[:500], file_arr[500:]
X_train, Y_train = file_train[:,:45], file_train[:,45:]
X_test, Y_test = file_test[:,:45], file_test[:,45:]
C_Grid = np.linspace(0.0001, 0.6, 100)

'''I created a dictionary, the key is "C", the value is the list of C_Grid'''
K=C_Grid.tolist()
param_grid = {'C':K}

'''I set cv to KFold(n_splits=10), and fix scoring to "neg_log_loss" '''
grid_lr = GridSearchCV(estimator=LogisticRegression(penalty='l1',
      solver='liblinear'), cv=KFold(n_splits=10), param_grid =param_grid, scoring = 'neg_log_loss',)
grid_lr.fit(X_train, Y_train.ravel())

'''Got the corresponding optimized C'''
grid_lr.best_params_