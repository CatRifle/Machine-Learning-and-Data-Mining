import os
from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets,linear_model,preprocessing
from sklearn.metrics import mean_squared_error, r2_score

file=pd.read_csv('/content/drive/MyDrive/data.csv')
file_rescale = preprocessing.scale(file)

data = file_rescale
X, y = data[:, :-1], data[:, -1]
Xt = X[1:]
yt = y[1:]

lam = ([np.log(0.01), np.log(0.1), np.log(0.5), np.log(1), np.log(1.5), np.log(2), np.log(5),
        np.log(10), np.log(20), np.log(30), np.log(50), np.log(100), np.log(200), np.log(300)])


coefs = []
for item in lam:
    	ridge = linear_model.Ridge(alpha=item, fit_intercept=True)    
    	ridge.fit(Xt, yt)
    	coefs.append(ridge.coef_)
        

Coefs = np.array(coefs)
y1=Coefs[:,0]
y2=Coefs[:,1]
y3=Coefs[:,2]
y4=Coefs[:,3]
y5=Coefs[:,4]
y6=Coefs[:,5]
y7=Coefs[:,6]
y8=Coefs[:,7]

ax = plt.gca()
Colr = ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey']
  
ax.plot(lam,y1,color= Colr[0])
ax.plot(lam,y2,color= Colr[1])
ax.plot(lam,y3,color= Colr[2])
ax.plot(lam,y4,color= Colr[3])
ax.plot(lam,y5,color= Colr[4])
ax.plot(lam,y6,color= Colr[5])
ax.plot(lam,y7,color= Colr[6])
ax.plot(lam,y8,color= Colr[7])

 
plt.xlabel('lambda')
plt.ylabel('coefs')
plt.title('Ridge coefficients')
plt.axis('tight')
plt.legend(('X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'),loc='upper right')
plt.show() 