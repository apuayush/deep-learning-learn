#other libraries
import pandas as pd
import numpy as np

#sklearn libraries
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, accuracy_score


class Network(object):
    def __init__(self, data_location):
        """
        starting the dataset location
        :param data_location:
        """
        self.dataset = np.array(pd.read_csv(data_location).iloc[:,:].values)
        self.X = None
        self.Y = None
        self.w = np.zeros((len(self.dataset[0]), 1))
        self.b = 0
        self.X_train = None
        self.X_test = None
        self.Y_test = None
        self.Y_train = None
        # print(self.X)
        # TODO - structure of Y could create a problem

    def transform(self):
        label_x1 = LabelEncoder()
        label_x2 = LabelEncoder()
        self.X[:, 1] = label_x1.fit_transform(self.X[:, 1])
        self.X[:, 2] = label_x2.fit_transform(self.X[:, 2])
        self.Y = label_x2.fit_transform(self.Y)
        onehot = OneHotEncoder(categorical_features=[1])
        self.X = onehot.fit_transform(self.X).toarray()
        self.X = self.X[:, 1:].T

    def data_segregation(self, test_percent=25, randomize=0):
        # shuffle the data if randomize is True
        if randomize:
            np.random.shuffle(self.dataset)

        # separate dependent and independent variables
        self.X = np.array(self.dataset[:, 3:13])
        self.Y = np.array(self.dataset[:, 13])
        self.transform()

        size = len(self.X[0])
        margin = int((1-test_percent/100)*size)
        self.X_train = self.X[:, 0:margin]
        self.X_test = self.X[:, margin:]
        self.Y_train = self.Y[:margin]
        self.Y_test = self.Y[margin:]

    def start(self):
        self.propagate()

    def propagate(self):
        m = self.X_train.shape[1]
        Y_ = sigmoid(np.dot(self.w.T, self.X_train)+self.b)
        cost = -1/m*np.sum(self.Y_train*np.log(Y_)+(1-self.Y_train)*np.log(1-Y_))

        dw = 1 / m * np.dot(self.X_train, (Y_ - self.Y_train).T)
        db = 1 / m * np.sum(Y_ - self.Y_train)

        assert (dw.shape == self.w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost


def sigmoid(z):
    return 1/(1+np.exp(-1*z))


if __name__ == "__main__":
    obj = Network("Churn_Modelling.csv")
    obj.data_segregation(randomize=1)
