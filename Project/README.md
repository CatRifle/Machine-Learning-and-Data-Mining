# COMP9417 2021 T2 Project

# CHALLENGE

- Hello, in this project, I plan to do the Kaggle featured prediction challenge “Amazon.com - Employee Access Challenge”. The challenge overview can be accessed here:
https://www.kaggle.com/c/amazon-employee-access-challenge/overview

# CSV FILE BRIG INTRODUCTION

- <b>train.csv:</b> The original data set for training and testing. It consists of 10 columns, 9 of which are features and 1 is label.
- <b>test.csv:</b> The data set for predicting final "ACTION". It consists of 10 columns, 9 of which are features and 1 is index.
- <b>sampleSubmission.csv:</b> The expected format of final output prediction.


# PY/IPYNB FILE BRIEF INTRODUCTION

- <b>LogisticRegression.py:</b> Prediction with LogisticRegression classifier
- <b>LinearRegression.py:</b> Prediction with LinearRegression classifier
- <b>DecisionTrees.py:</b> Prediction with DecisionTree classifier
- <b>RandomForset.py:</b> Prediction with RandomForest classifier
- <b>Catboost.py:</b> Prediction with Catboost classifier (Please pip install/upgrade catboost in your computer to ensure later "import catboost" function, thanks)
- <b>ShapVisualization.py:</b> Use shap function to visualize each feature distribution and correlation (Please pip install/upgrade shap in your computer, thanks)
- <b>9417_proj.ipynb:</b> This is the whole process of the project
  
  # CONCLUSION
  
- Catboost is the better classifer for this challenge
- When the parameter cat_features is set to a list consists of 9 features, the model reach its optimization. 
