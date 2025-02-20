import numpy as np
import pandas as pd

TEST_DATA_COUNT = 2000


def get_data():
    data = pd.read_csv("data/train.csv").to_numpy()
    np.random.shuffle(data)

    train_data = data[TEST_DATA_COUNT:].T
    test_data = data[:TEST_DATA_COUNT].T

    X_train = train_data[1:]
    y_train = train_data[0]

    X_test = test_data[1:]
    y_test = test_data[0]

    return X_train, y_train, X_test, y_test


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    Z_max = np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_prop(X, W1, b1, W2, b2):
    Z1 = np.matmul(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.matmul(W2, A1) + b2
    A2 = softmax(Z2)

    return A2


def main():
    X_train, y_train, X_test, y_test = get_data()
    W1, b1, W2, b2 = init_params()
    A2 = forward_prop(X_train, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
