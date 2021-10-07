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
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

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

'''Predict function'''
def predict(W, inputs):
    return jnp.dot(inputs, W)

'''Loss function'''
def Loss_new(W):
  pred = predict(W, X_new)
  inner_sqrt = 1/4 *jnp.square(Y_train-pred)+1.0
  mean_Los = jnp.mean(jnp.sqrt(inner_sqrt)-1.0)
  return mean_Los

'''Previous X was n x 3, now X_new we need is n x 4 format. Insert 1.0 to the first index of each X vector'''
X_new=[]
for i in range(len(X_train)):
    x = X_train[i]
    Lx = list(x)
    Lx.insert(0,1.0)
    x = jnp.array(Lx)
    X_new.append(x)
X_new = jnp.array(X_new)

'''Calculate list of loss and weight'''
diff_loss = 1
List_W = []
List_loss = []
W = jnp.array([1.0,1.0,1.0,1.0])
List_W.append(W)
List_loss.append(Loss_new(W))
k = 0
while diff_loss>=0.0001:
  loss1 = Loss_new(W)
  W_grad = grad(Loss_new)(W)
  W = W - W_grad
  List_W.append(W)
  loss2 = Loss_new(W)
  List_loss.append(loss2)
  diff_loss = abs(loss2-loss1)
  k += 1
  
''' Use the final W for test validation'''
X_test_new=[]
for i in range(len(X_test)):
    x = X_test[i]
    Lx = list(x)
    Lx.insert(0,1)
    x = jnp.array(Lx)
    X_test_new.append(x)
X_test_new = jnp.array(X_test_new)
pred = predict(W, X_test_new)
Last_Test_Loss = jnp.mean(jnp.sqrt(1/4 *jnp.square(Y_test-pred)+1.0)-1.0)
last_test_loss= float(Last_Test_Loss)

'''Plot the curve of the loss change'''
print("It took",len(List_loss),"iterations to converge the loss.")
plt.plot(List_loss)
plt.show()

'''Print the Final train loss and test loss'''
print("Final train loss:", List_loss[-1])
print("Final test loss: ", last_test_loss)
print("Final weight vector:",List_W[-1])