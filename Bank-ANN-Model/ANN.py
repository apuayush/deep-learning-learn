# libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# keras libraries
from keras.models import Sequential
from keras.layers import Dense

np.set_printoptions(threshold=np.nan)

# 0-import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,13].values

label_x1 = LabelEncoder() # 0 for france 1 for germany and so on
label_x2 = LabelEncoder() 

X[:, 1] = label_x1.fit_transform(X[:, 1])
X[:, 2] = label_x2.fit_transform(X[:, 2])
Y = label_x2.fit_transform(Y)
onehot = OneHotEncoder(categorical_features = [1]) #encoding each data with dummy variables only for countries
X = onehot.fit_transform(X).toarray() 
X = X[:,1:]
# 1- segregation of test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)
# feature scaling(standardization)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Initialising the ANN
# output sigmoid
# hidden layer rectifier

classifier = Sequential()
# hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))

# 2nd hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))

# output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
# set your own classifier models linear, multiple or logistic regression

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
# adam is an stochastic gradient descent algorithm
# loss = sqrt((x2-x1)^2 + (y2-y1)^2) for linear and here logarithmic

classifier.fit(X_train, Y_train, batch_size = 10, epochs = 150)

# Y_pred vector to store prediction
# predict the test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)



# using confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
