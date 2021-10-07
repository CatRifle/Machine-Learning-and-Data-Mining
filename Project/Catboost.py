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
Sample = pd.read_csv('/content/drive/MyDrive/9417 Proj/sampleSubmission.csv')

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

"I have pip installed catboost in my file, so I just import catboost here"
import catboost
from catboost.eval.evaluation_result import *
from catboost import CatBoostClassifier,CatBoostRegressor, Pool, MetricVisualizer

Model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", verbose=200, random_seed=1)
Model.fit(Train_X, Train_Y, eval_set=(Valid_X, Valid_Y),plot=True,use_best_model=True)
Pred_Y = Model.predict(Valid_X1)
print(classification_report(Valid_Y1, Pred_Y))
print("Train Set Accuracy_Score:",end="")
print(Model.score(Train_X, Train_Y))
print("Validation Accuracy_Score:",end="")
print(Model.score(Valid_X, Valid_Y))

Model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", verbose=200, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8], random_seed=1)
Model.fit(Train_X, Train_Y, eval_set=(Valid_X, Valid_Y), use_best_model=True, plot=True)
Pred_Y = Model.predict(Valid_X1)
print(classification_report(Valid_Y1, Pred_Y))
print("Train Set Accuracy_Score:",end="")
print(Model.score(Train_X, Train_Y))
print("Validation Accuracy_Score:",end="")
print(Model.score(Valid_X, Valid_Y))

'''
fetch the feature importance and print out'''
f_importance = Model.get_feature_importance(prettified=True)
print(f_importance)

"plot the feature importance histogram"
sns.barplot(x='Importances', y='Feature Id', data=f_importance, saturation=4.0)

Sample1= Sample.iloc[100:105,:]
print(Sample1)

Pred_Test_Y = Model.predict(Test_X)
type(Pred_Test_Y)

'''
Predict the action of test set
Save the prediction result of Test_X set as 'Employee_Access_Predict.csv' '''
Predictive_Model = pd.DataFrame({
        "Id": Column_ID,
        "Action": Pred_Test_Y
})
Predictive_Model.to_csv('Employee_Access_Predict.csv', index=False)

