import numpy as np
import pandas as pd

TEST_DATA_COUNT = 2000


def get_data():
    data = pd.read_csv("data/train.csv").to_numpy()
    np.random.shuffle(data)

    train_data = data[TEST_DATA_COUNT:].T
    test_data = data[:TEST_DATA_COUNT].T

    x_train = train_data[1:]
    y_train = train_data[0]

    x_test = test_data[1:]
    y_test = test_data[0]


def main():
    get_data()


if __name__ == "__main__":
    main()
