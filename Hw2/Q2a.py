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


A= np.array([[1,0,1,-1],[-1,1,0,2],[0,-1,-2,1]])
B= np.array([[1],[2],[3]])

def Derv_F(x):
  Vect1 = A.dot(x)-B
  A_transpose = A.T
  Vect2 = A_transpose.dot(Vect1)
  x_current = x-0.1*Vect2
  return Vect2, x_current


X=np.array([[1],[1],[1],[1]])
Norm2_derv=1
X_array=[[1,1,1,1]]
while Norm2_derv>=0.001:
  derv= Derv_F(X)[0]
  Norm2_derv = 0
  for i in range(4):
    Norm2_derv = Norm2_derv + derv[i]*derv[i]
  Norm2_derv = math.sqrt(Norm2_derv)
  X = Derv_F(X)[1]
  T = []
  for j in X:
    T.append(float(j))
  X_array.append(T)
  
'''Print out length of X_array'''
print(len(X_array))

'''Print out first 5 rows of vector X'''
for i in range(5):
  print("k = " ,end = "")
  print(i, end=",  x")
  print(f"({i}) = ", end="")
  print(X_array[i])
  
'''Print out last 5 rows of vector X'''
for i in range(219,224):
  print("k = " ,end = "")
  print(i, end=",  x")
  print(f"({i}) = ", end="")
  print(X_array[i])