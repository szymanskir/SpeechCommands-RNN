from keras import models, layers
from typing import Tuple


def create_sample_rnn(input_shape: Tuple[int], num_classes):
    model = models.Sequential()
    model.add(layers.CuDNNLSTM(32, input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
