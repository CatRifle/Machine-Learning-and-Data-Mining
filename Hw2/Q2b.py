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


'''Define a function to update X and alpha'''
def Derv_Fb(x,a,i):
  Vect0 = (A.dot(x)).T
  Vect1 = A.dot(x)-B
  A_transpose = A.T
  Vect2 = A_transpose.dot(Vect1)
  A2 = A.dot(A_transpose)
  Vect3 = (B.T).dot(A2)
  Vect4 = A2.dot(Vect1)
  Hx = A.dot(Vect2)

  Nume1= (A.dot(x)).T
  Nume2= B.T
  Numerator = Nume1.dot(Hx) - Nume2.dot(Hx)
  Denominator = (Hx.T).dot(Hx)
  a = Numerator/Denominator
  if i ==0 :
    x = x- 0.1*Vect2
  else:
    x = x- a*Vect2
  return Vect2, x, a

"Do the steepest gradient descent, record updated X and alpha"
X=np.array([[1],[1],[1],[1]])
Norm2_derv=1
A_arr =[]
X_array=[[1,1,1,1]]
n=0
insta_a = 0.1
A_arr.append(insta_a)
while Norm2_derv>=0.001:
  List = Derv_Fb(X,insta_a,n)
  derv= List[0]
  X = List[1]
  insta_a = List[2]
  T = []
  Norm2_derv = 0
  for i in range(4):
    Norm2_derv = Norm2_derv + derv[i]*derv[i]
  Norm2_derv = math.sqrt(Norm2_derv)

  if n>0:
    A_arr.append(float(insta_a))
  if n>0:
    for j in X:
      T.append(float(j))
    X_array.append(T)
  n=n+1
  
  
'''Plot figure of A_array'''  
print(len(A_arr))
plt.plot(A_arr)
plt.show()

'''Print out first 5 rows of vector X, and alpha '''
for i in range(5):
  print("k = " ,end = "")
  print(i, end=",  a")
  print(f"({i}) = ", end="")
  print('{:f}'.format(A_arr[i]), end="")

  print(i, end=",  x")
  print(f"({i}) = ", end="")
  print(X_array[i])
  

'''Print out last 5 rows of vector X, and alpha '''    
for i in range(87,91):
  print("k = " ,end = "")
  print(i, end=",  a")
  print(f"({i}) = ", end="")
  print('{:f}'.format(A_arr[i]), end="")

  print(i, end=",  x")
  print(f"({i}) = ", end="")
  print(X_array[i])



