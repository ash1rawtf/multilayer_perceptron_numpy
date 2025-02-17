import numpy as np
from numpy.typing import NDArray
import pandas as pd
import typing

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


def main():
    X_train, y_train, X_test, y_test = get_data()
    W1, b1, W2, b2 = init_params()


if __name__ == "__main__":
    main()
