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

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    return X_train, X_test, Y_train, Y_test
