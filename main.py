from transformer import Transformer
import tensorflow as tf
import numpy as np
from get_data import get_data
from train import train


def main():
    # x = np.array([[[0.5, 0.8, 0.5, 1.7, 2.2, 0.5], [1.2, 3.2, 5, 8.1, 1.2, 0.3]],
    #               [[1.5, 1.8, 1.5, 2.7, 3.2, 1.5], [2.2, 4.2, 6, 9.1, 1.2, 1.3]],
    #               [[2.5, 2.8, 2.5, 3.7, 4.2, 2.5], [3.2, 5.2, 7, 10.1, 2.2, 2.3]]])
    # x = x.reshape([2, 6, 3])
    # y = np.array([[[1.2, 1.5, 1.6], [2.5, 2.6, 3.8]],
    #               [[2.2, 2.5, 2.6], [3.5, 3.6, 4.8]],
    #               [[3.2, 3.5, 3.6], [4.5, 4.6, 5.8]]])
    # y = y.reshape([2, 3, 3])
    train_dataset = get_data()

    train(train_dataset)


if __name__ == '__main__':
    main()