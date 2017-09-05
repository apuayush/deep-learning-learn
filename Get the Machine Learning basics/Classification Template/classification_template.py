# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#graph lib
from matplotlib.colors import ListedColormap

np.set_printoptions(threshold=np.nan)

def data_prediction():
    # 0-import dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, 2:4].values
    Y = dataset.iloc[:,4].values
    
    # 1- segregation of test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25,random_state = 0)
    
    # feature scaling(standardization)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    # fitting logistic regression to the training set
    classifier = LogisticRegression(random_state = 0) # random_state = 0 to set no shuffling
    classifier.fit(X_train, Y_train)
    
    # predict the test set results
    Y_pred = classifier.predict(X_test)
    # Y_pred vector to store prediction
    
    # using confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    # training graph
    plot_graph(X_train, Y_train, classifier, 'train_data')
    
    #test graph
    plot_graph(X_test, Y_test, classifier, 'test_data')

def plot_graph(X_set, Y_set, classifier, title):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                         np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([
            X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i,j in enumerate(np.unique(Y_set)):
        plt.scatter(X_set[Y_set==j, 0], X_set[Y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i),label = j)
    
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated salary')
    plt.legend()
    plt.show()
    
    
data_prediction()