from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
import sklearn
import math
import os
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import datasets,linear_model,preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge

""" Hello, tutor. I used google colab to train and test, I imported the csv file from google drive
    if you want to test the .csv on you pc, you probably need to 
    change the file path  , thanks
"""

from google.colab import drive
drive.mount('/content/drive')
Train = pd.read_csv('/content/drive/MyDrive/9417 Proj/train.csv')
Test = pd.read_csv('/content/drive/MyDrive/9417 Proj/test.csv')


print(Train.head())
print("Train Set Shape:",end="")
print(Train.shape)

print(Test.head())
print("Test Set Shape:",end="")
print(Train.shape)

plt.figure(figsize=(10, 8))
sns.heatmap(Train.corr(), annot=True,cmap="GnBu_r")

'''Split Train into Train_Y and Train_X
  Train_Y is the first column 'ACTION'
  Train_X is the combination of other columns
  Test_X is the combination after Test drops column 'id' 
'''
Train_Y = Train["ACTION"]
Train_X = Train.drop("ACTION",axis=1)
Column_ID = Test["id"]
Test_X = Test.drop("id",axis=1)


Train_X, Valid_X, Train_Y, Valid_Y = split(Train_X, Train_Y, test_size=0.2, random_state=0, stratify=Train_Y)

print("Train set after pre-processing:",end="")
print(Train_X.head())
print("Test set after pre-processing:",end="")
print(Test_X.head())

Train_X1, Train_Y1 = np.array(Train_X), np.array(Train_Y)
Valid_X1, Valid_Y1 = np.array(Valid_X), np.array(Valid_Y)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(Train_X1, Train_Y1)
Pred_Y = RF.predict(Valid_X1)
print("Random Forest")
print(classification_report(Valid_Y1, Pred_Y))

Pred = []
for i in Pred_Y:
  if i>= 0.5:
    Pred.append(1)
  else :
    Pred.append(0)
Pred = np.array(Pred)

'''
The prediction labels are probabilties
need to change it into label 0 or 1 '''

print("Accuracy_Score:",end="")
Accuracy1 = accuracy_score(Pred, Valid_Y1)
print(Accuracy1)

'''
fetch the feature importance and print out'''
impt = RF.feature_importances_
Fea_list = Train_X.columns
f_importance = pd.DataFrame({
        "Features": Fea_list,
        "Importance": impt 
})
f_importance= f_importance.sort_values(by=["Importance"],ascending=False)
print(f_importance)

"plot the feature importance histogram"
sns.barplot(x='Importance', y="Features", data=f_importance, saturation=2.0 )