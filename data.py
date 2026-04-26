import numpy as np
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y = np.eye(3)[y]

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X, Y = X[indices], Y[indices]

    X_train, X_test = X[:120], X[120:]
    Y_train, Y_test = Y[:120], Y[120:]

    return X_train, X_test, Y_train, Y_test
