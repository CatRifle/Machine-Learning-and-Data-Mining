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
sns.pairplot(file)
plt.show()