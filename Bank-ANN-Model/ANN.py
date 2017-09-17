# libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

# keras libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

np.set_printoptions(threshold=np.nan)

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,13].values

label_x1 = LabelEncoder() 
label_x2 = LabelEncoder() 

X[:, 1] = label_x1.fit_transform(X[:, 1])
X[:, 2] = label_x2.fit_transform(X[:, 2])
Y = label_x2.fit_transform(Y)
onehot = OneHotEncoder(categorical_features = [1])
X = onehot.fit_transform(X).toarray() 
X = X[:,1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
        'batch_size': [25,32],
        'nb_epoch': [100, 500],
        'optimizer': ['rmsprop','adam']
        }
ftwo_score = ['accuracy','precision']
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring=ftwo_score,
                           n_jobs=-1, 
                           cv = 10)
grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#accuracy = cross_val_score(estimator=classifier,X=X_train, y=Y_train, cv=10,n_jobs=-1 )
#mean = accuracy.mean()
#variance = accuracy.std()

#Y_pred = classifier.predict(X_test)
#Y_pred = (Y_pred > 0.5)
#cm = confusion_matrix(Y_test, Y_pred)