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

file_array= np.genfromtxt('/content/drive/MyDrive/9417 HW2/Q1.csv',delimiter=",")

file_arr = file_array[1:]
file_train, file_test = file_arr[:500], file_arr[500:]
X_train, Y_train = file_train[:,:45], file_train[:,45:]
X_test, Y_test = file_test[:,:45], file_test[:,45:]
C_Grid = np.linspace(0.0001, 0.6, 100)

L=[]
for item in C_Grid:
  Clf = sklearn.linear_model.LogisticRegression(C=item, solver='liblinear',penalty='l1')
  T=[]
  for t in range(10):   
    a, b = 50*t, 50*t+50
    current_X_test = X_train[a:b,:]
    current_Y_test = Y_train[a:b,:]
    current_X_train = np.vstack((X_train[0:a,:], X_train[b:500,:])) 
    current_Y_train = np.vstack((Y_train[0:a,:], Y_train[b:500,:]))
    Clf.fit(current_X_train, current_Y_train.ravel())
    pred_Y = Clf.predict_proba(current_X_test)
    loss = sklearn.metrics.log_loss(current_Y_test, pred_Y)
    T.append(loss)
  L.append(T)
  
Loss_mean=[]
for i in range(100):
  mean = sum(L[i])/10
  Loss_mean.append(mean)

print(min(Loss_mean))
print(Loss_mean.index(min(Loss_mean)))

"After calculation, I got index= 31"
opti_C= C_Grid[31]
print(opti_C)
"opti_C = 0.18794747474747472"

Clf = sklearn.linear_model.LogisticRegression(C=opti_C, solver='liblinear',penalty='l1')
Clf.fit(X_train, Y_train)
pred_train_Y = Clf.predict_proba(X_train)
pred_test_Y = Clf.predict_proba(X_test)

Accuracy_Train = Clf.score(X_train, Y_train)
Accuracy_Test = Clf.score(X_test, Y_test)
print(Accuracy_Train)
print(Accuracy_Test)

"After calculation"
"Accuracy_Train=0.752"
"Accuracy_Test=0.74"

plt.xlabel('C_Grid')  
plt.ylabel('log_loss')  
plt.xticks(np.linspace(0.0001, 0.6, 100))

"Actually, we don't need to have xticks because it's dense and we can't split 100 number away in such a plot"
ax = sns.boxplot(data=L)
plt.show()
