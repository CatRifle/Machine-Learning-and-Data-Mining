import os
from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets,linear_model,preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error,mean_absolute_error 
from sklearn.model_selection import LeaveOneOut
method = LeaveOneOut()

file=pd.read_csv('/content/drive/MyDrive/data.csv')
file_rescale = preprocessing.scale(file)

data = file_rescale
X, y = data[:, :-1], data[:, -1]
Xt = X[1:]
yt = y[1:]

from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error,mean_absolute_error 
X = Xt
Y = yt
lamdda = np.arange(0, stop=50.1, step=0.1)

coefs_Lasso = []
MSE =list()
for item in lamdda:
   Y_true, Y_predict = list(), list()
   ERR = list(); MERR = 0; 
   for train_ix, test_ix in method.split(X):
     # split data
     X_train, X_test = X[train_ix, :], X[test_ix, :]
     y_train, y_test = Y[train_ix], Y[test_ix]
     lasso = linear_model.Lasso(alpha=item, fit_intercept=True)    
     lasso.fit(X_train, y_train)
  
     yhat = lasso.predict(X_test)
     Y_true.append(y_test[0])
     Y_predict.append(yhat[0])
     er = abs(y_test[0]-yhat[0])
     ERR.append(er)
   for i in ERR:
     MERR = MERR + i*i
      
   MSE.append(MERR)
   coefs_Lasso.append(lasso.coef_)
print(len(MSE))
print(MSE.index(min(MSE)))


Coefs_Lasso = np.array(coefs_Lasso)
y1=Coefs_Lasso[:,0]
y2=Coefs_Lasso[:,1]
y3=Coefs_Lasso[:,2]
y4=Coefs_Lasso[:,3]
y5=Coefs_Lasso[:,4]
y6=Coefs_Lasso[:,5]
y7=Coefs_Lasso[:,6]
y8=Coefs_Lasso[:,7]
ax = plt.gca()
Colr = ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey']
  
ax.plot(lamdda,y1,color= Colr[0])
ax.plot(lamdda,y2,color= Colr[1])
ax.plot(lamdda,y3,color= Colr[2])
ax.plot(lamdda,y4,color= Colr[3])
ax.plot(lamdda,y5,color= Colr[4])
ax.plot(lamdda,y6,color= Colr[5])
ax.plot(lamdda,y7,color= Colr[6])
ax.plot(lamdda,y8,color= Colr[7])

plt.xlabel('lambda')
plt.ylabel('coefs')
plt.title('Loocv Lasso coefficients')
plt.axis('tight')
plt.legend(('X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'),loc='upper right')
plt.show() 