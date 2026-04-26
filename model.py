import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, h_dim, out_dim, lr=0.05):
        self.lr = lr

        # He initialization
        self.W1 = np.random.randn(input_dim, h_dim) * np.sqrt(2 / input_dim)
        self.B1 = np.zeros((1, h_dim))

        self.W2 = np.random.randn(h_dim, out_dim) * np.sqrt(2 / h_dim)
        self.B2 = np.zeros((1, out_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.A1 = X @ self.W1 + self.B1
        self.H1 = self.relu(self.A1)

        self.A2 = self.H1 @ self.W2 + self.B2
        self.Z = self.softmax(self.A2)

        return self.Z

    def loss(self, Y, Z):
        return np.mean(-np.sum(Y * np.log(Z + 1e-15), axis=1))

    def backward(self, X, Y):
        m = X.shape[0]

        dA2 = self.Z - Y
        dW2 = (self.H1.T @ dA2) / m
        dB2 = np.sum(dA2, axis=0, keepdims=True) / m

        dH1 = dA2 @ self.W2.T
        dA1 = dH1 * (self.A1 > 0)

        dW1 = (X.T @ dA1) / m
        dB1 = np.sum(dA1, axis=0, keepdims=True) / m

        return dW1, dW2, dB1, dB2

    def update(self, dW1, dW2, dB1, dB2):
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.B1 -= self.lr * dB1
        self.B2 -= self.lr * dB2

    def predict(self, X):
        Z = self.forward(X)
        return np.argmax(Z, axis=1)

    def accuracy(self, X, Y):
        preds = self.predict(X)
        y_true = np.argmax(Y, axis=1)
        return np.mean(preds == y_true)
