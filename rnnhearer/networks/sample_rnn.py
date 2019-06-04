from keras import models, layers
from typing import Tuple
from .network_configuration import NetworkConfiguration
from keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dense,
    Activation,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
)


def create_sample_rnn(input_shape: Tuple[int], num_classes):
    model = models.Sequential()
    model.add(layers.CuDNNLSTM(32, input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def create_lstm_network_from_config(
    network_configuration: NetworkConfiguration,
    input_shape: Tuple[int],
    num_classes: int,
):
    model = models.Sequential(name=network_configuration.name)
    should_return_sequence = [
        True for x in range(len(network_configuration.units_per_layer))
    ]
    should_return_sequence[-1] = False
    model.add(
        layers.LSTM(
            units=network_configuration.units_per_layer[0],
            input_shape=input_shape,
            dropout=network_configuration.dropout_probabilities[0],
            recurrent_dropout=network_configuration.recurrent_dropout_probabilities[0],
            return_sequences=should_return_sequence[0],
        )
    )

    layers_count = len(network_configuration.units_per_layer)
    for layer_num in range(1, layers_count):
        layer = layers.LSTM(
            units=network_configuration.units_per_layer[layer_num],
            dropout=network_configuration.dropout_probabilities[layer_num],
            recurrent_dropout=network_configuration.recurrent_dropout_probabilities[
                layer_num
            ],
            return_sequences=should_return_sequence[layer_num],
        )
        model.add(layer)
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def create_cnn_network_from_config(
    network_configuration: NetworkConfiguration,
    input_shape: Tuple[int],
    num_classes: int,
):
    model = models.Sequential()
    model.add(Conv1D(16, kernel_size=3, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2, padding="same"))
    for i in range(1, 8):
        model.add(Conv1D(16 * (2 ** i), kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling1D(2, padding="same"))

    # model.add(GlobalAveragePooling1D())
    model.add(GlobalMaxPooling1D())

    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    return model
