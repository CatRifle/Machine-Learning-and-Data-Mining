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

def Calculate_SquareSum(df,num):
  s_square = 0
  for i in file_rescale:
    item = i[num-1]
    s_square = s_square + item*item
  print("The Sum of squared observations of feature X%d is " % num, end="");
  print("{:.2f}".format(s_square))
  return 


Calculate_SquareSum(file_rescale,1)
Calculate_SquareSum(file_rescale,2)
Calculate_SquareSum(file_rescale,3)
Calculate_SquareSum(file_rescale,4)
Calculate_SquareSum(file_rescale,5)
Calculate_SquareSum(file_rescale,6)
Calculate_SquareSum(file_rescale,7)
Calculate_SquareSum(file_rescale,8)