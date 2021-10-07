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

for i in Train_X.describe().columns:
    sns.boxplot(Train_X[i].dropna(), color='g')
    plt.show()

import catboost
from catboost.eval.evaluation_result import *
from catboost import CatBoostClassifier, Pool, MetricVisualizer

P1 = Pool(data=Train_X, label=Train_Y, cat_features=[0,1,2,3,4,5,6,7,8])
values = Model.get_feature_importance(data=P1, type='ShapValues')
expected_val = values[1,-1]
shap_val = values[:,:-1]

"Smaple index 1"
import shap
shap.initjs()
shap.force_plot(expected_val, shap_val[1,:], Train_X.iloc[1,:])

"Smaple index 50"
import shap
shap.initjs()
values = Model.get_feature_importance(data=P1, type='ShapValues')
expected_val = values[50,-1]
shap_val = values[:,:-1]
shap.force_plot(expected_val, shap_val[50,:], Train_X.iloc[50,:])

import shap
shap.initjs()
values = Model.get_feature_importance(data=P1, type='ShapValues')
shap_val = values[:,:-1]
shap.summary_plot(shap_val, Train_X, plot_type="bar")
shap.summary_plot(shap_val, Train_X)

shap.dependence_plot("RESOURCE", shap_values, Train_X, interaction_index=None)
shap.dependence_plot("MGR_ID", shap_values, Train_X, interaction_index=None)
shap.dependence_plot("ROLE_DEPTNAME", shap_values, Train_X, interaction_index=None)
shap.dependence_plot("ROLE_TITLE", shap_values, Train_X,interaction_index=None)
shap.dependence_plot("ROLE_FAMILY_DESC", shap_values, Train_X, interaction_index=None)

import shap
shap.initjs()
x_small = Train_X.iloc[0:200]
shap_small = shap_values[:200]
shap.force_plot(expected_value, shap_small, x_small)