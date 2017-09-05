#This is for including data and handling missing data

# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

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
label_x = LabelEncoder() # 0 for france 1 for germany and so on
label_y = LabelEncoder() # 1 for yes and 0 for no
X[:, 0] = label_x.fit_transform(X[:, 0])
Y = label_y.fit_transform(Y)
onehot = OneHotEncoder(categorical_features = [0]) #encoding each data with dummy variables 
X = onehot.fit_transform(X).toarray() 

# dividing data into test and training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)