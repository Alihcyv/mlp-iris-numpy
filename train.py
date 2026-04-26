import numpy as np
from model import NeuralNetwork
from data import load_data

def train():
    input_dim = 4
    h_dim = 10
    out_dim = 3
    batch_size = 16
    epochs = 100
    lr = 0.05

    X_train, X_test, Y_train, Y_test = load_data()

    model = NeuralNetwork(input_dim, h_dim, out_dim, lr)

    for epoch in range(epochs):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        X_train = X_train[indices]
        Y_train = Y_train[indices]

        losses = []

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            Z = model.forward(X_batch)
            loss = model.loss(Y_batch, Z)

            grads = model.backward(X_batch, Y_batch)
            model.update(*grads)

            losses.append(loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Loss: {np.mean(losses):.4f} | "
                f"Train Acc: {model.accuracy(X_train, Y_train):.4f} | "
                f"Test Acc: {model.accuracy(X_test, Y_test):.4f}"
            )

if __name__ == "__main__":
    train()
