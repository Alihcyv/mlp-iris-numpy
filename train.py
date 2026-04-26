import numpy as np
from sklearn import datasets

input_dim, h_dim, out_dim = 4, 10, 3  
batch_size = 16
num_epoch = 100
alpha = 0.05

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def to_one_hot(y, out_dim):
    return np.eye(out_dim)[y]

Y_oh = to_one_hot(y, out_dim)

indices = np.arange(len(X))
np.random.shuffle(indices)
X, Y_oh = X[indices], Y_oh[indices]

X_train, y_train = X[:120], Y_oh[:120]
X_test, y_test = X[120:], Y_oh[120:]

W1 = np.random.randn(input_dim, h_dim) * np.sqrt(2/input_dim)
B1 = np.zeros((1, h_dim))
W2 = np.random.randn(h_dim, out_dim) * np.sqrt(2/h_dim)
B2 = np.zeros((1, out_dim))

def ReLU(X):
    return np.maximum(X, 0)

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def loss(Y, Z):
    return np.mean(-np.sum(Y * np.log(Z + 1e-15), axis=1))

def feedforward(W1, W2, B1, B2, X):
    A1 = X @ W1 + B1
    H1 = ReLU(A1)
    A2 = H1 @ W2 + B2
    Z = softmax(A2)
    return A1, H1, A2, Z

def Backward(A1, H1, A2, Z, Y, X):
    m = X.shape[0] 
    
    dE_dA2 = Z - Y         
    dE_dW2 = (H1.T @ dE_dA2) / m  
    dE_dB2 = np.sum(dE_dA2, axis=0, keepdims=True) / m      
    
    dE_dH1 = dE_dA2 @ W2.T   
    dE_dA1 = dE_dH1 * (A1 > 0)
    
    dE_dW1 = (X.T @ dE_dA1) / m
    dE_dB1 = np.sum(dE_dA1, axis=0, keepdims=True) / m       
    return dE_dW2, dE_dW1, dE_dB2, dE_dB1

def Update(dW2, dW1, dB2, dB1, W1, W2, B1, B2):
    W1 -= alpha * dW1
    W2 -= alpha * dW2
    B1 -= alpha * dB1
    B2 -= alpha * dB2
    return W1, W2, B1, B2

def predict(X):
    _, _, _, Z = feedforward(W1, W2, B1, B2, X)
    return np.argmax(Z, axis=1)

def accuracy(x_data, y_data):
    prediction = predict(x_data)
    y_true = np.argmax(y_data, axis=1)
    return np.mean(prediction == y_true)

for epoch in range(num_epoch):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    
    epoch_losses = [] 
      
    for i in range(0, len(X_train), batch_size):
        X_sample = X_train[i:i+batch_size]
        Y_sample = y_train[i:i+batch_size]
        
        A1, H1, A2, Z = feedforward(W1, W2, B1, B2, X_sample)
        E = loss(Y_sample, Z)
        dW2, dW1, dB2, dB1 = Backward(A1, H1, A2, Z, Y_sample, X_sample)
        W1, W2, B1, B2 = Update(dW2, dW1, dB2, dB1, W1, W2, B1, B2)
        
        epoch_losses.append(E)

    if (epoch + 1) % 10 == 0:
        acc_tr = accuracy(X_train, y_train)
        acc_ts = accuracy(X_test, y_test)
        print(f"Epoch: {epoch+1:3} | Loss: {np.mean(epoch_losses):.4f} | Acc Train: {acc_tr:.4f} | Acc Test: {acc_ts:.4f}")
