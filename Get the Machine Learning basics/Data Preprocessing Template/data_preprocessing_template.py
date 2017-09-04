# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

np.set_printoptions(threshold=np.nan)

# import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values

# filling missing data with the mean of column
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data 
label_x = LabelEncoder()
label_y = LabelEncoder()
X[:, 0] = label_x.fit_transform(X[:, 0])
Y = label_y.fit_transform(Y)
onehot = OneHotEncoder(categorical_features = [0])
X = onehot.fit_transform(X).toarray()