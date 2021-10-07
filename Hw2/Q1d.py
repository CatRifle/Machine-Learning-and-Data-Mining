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

"""Hello, tutor. I used google colab to train and test, I imported the Q1.csv file from google drive
    if you want to test the .csv on you pc, you probably need to 
   change the file path and delete the command like drive.mount above, thanks
"""

file_array= np.genfromtxt('/content/drive/MyDrive/9417 HW2/Q1.csv',delimiter=",")
file_arr = file_array[1:]
file_train, file_test = file_arr[:500], file_arr[500:]
X_train, Y_train = file_train[:,:45], file_train[:,45:]
X_test, Y_test = file_test[:,:45], file_test[:,45:]
C_Grid = np.linspace(0.0001, 0.6, 100)

model = sklearn.linear_model.LogisticRegression(C=1, solver='liblinear',penalty='l1')
np.random.seed(12)

'''Iterate 10000 times, each time use the model to train the dataset we emerged with random function'''
Coe=[]
for i in range(10000):
  train_x_cr, train_y_cr =[], []
  for j in range(500):
    id= np.random.randint(0,499)
    train_x_cr.append(X_train[id])
    train_y_cr.append(Y_train[id].ravel())
  train_y_cr = np.array(train_y_cr)  
  model.fit(train_x_cr, train_y_cr.ravel())
  coefficient = model.coef_
  coeffs = coefficient.tolist()
  coeff = coeffs[0]
  Coe.append(coeff)

'''Calculate the coef of position 5% and 95%, and store them into lists'''
Arr_Coe = np.array(Coe)
Boundary_Up=[]
Boundary_Down=[]
for i in range(45):
  Single_Boundary_up =[]
  Single_Boundary_down =[]
  T= Arr_Coe[:,i]
  T.sort()
  percent_5 = T[500]
  percent_95 = T[9500]
  Boundary_Down.append(percent_5)
  Boundary_Up.append(percent_95)
  
'''Store 45 mean coefs'''
Mean=[]
for i in range(45):
  summ = sum(Arr_Coe[:,i])
  mean = summ/10000
  Mean.append(mean)
  
'''Store 45 bar heights'''
H =[]
for i in range(45):
  h= Boundary_Up[i]-Boundary_Down[i]
  H.append(h)
  

'''Use conditional sentence to determine red bar or blue bar'''
data= Boundary_Up
size = 45
x = np.arange(size)
'''Plot the bar chart with  matplotlib'''
for i in range(45):
  if Boundary_Down[i]*Boundary_Up[i]>0:
    plt.bar(i, height=H[i],bottom=Boundary_Down[i], label='b', color="b")
  else:
    plt.bar(i, height=H[i],bottom=Boundary_Down[i], label='b', color="r")
  plt.plot(i,Mean[i],"_")
plt.show()


