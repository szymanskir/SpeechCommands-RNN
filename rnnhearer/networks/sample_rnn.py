from keras import models, layers
from typing import Tuple
from .network_configuration import NetworkConfiguration


def create_sample_rnn(input_shape: Tuple[int], num_classes):
    model = models.Sequential()
    model.add(layers.CuDNNLSTM(32, input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def create_network_from_config(
    network_configuration: NetworkConfiguration,
    input_shape: Tuple[int],
    num_classes: int,
):
    model = models.Sequential(name=network_configuration.name)
    should_return_sequence = [
        True for x in range(len(network_configuration.units_per_layer))
    ]
    should_return_sequence[-1] = False

    if network_configuration.conv_layers_count > 0:
        model.add(
            layers.Conv2D(
                filters=1,
                kernel_size=(3, 3),
                padding="same",
                input_shape=(input_shape[0], input_shape[1], 1),
            )
        )
        for conv_layer_num in range(1, network_configuration.conv_layers_count):
            model.add(layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same"))
        model.add(layers.Reshape(target_shape=input_shape))
        model.add(
            layers.LSTM(
                units=network_configuration.units_per_layer[0],
                dropout=network_configuration.dropout_probabilities[0],
                recurrent_dropout=network_configuration.recurrent_dropout_probabilities[
                    0
                ],
                return_sequences=should_return_sequence[0],
            )
        )
    else:
        model.add(
            layers.LSTM(
                units=network_configuration.units_per_layer[0],
                input_shape=input_shape,
                dropout=network_configuration.dropout_probabilities[0],
                recurrent_dropout=network_configuration.recurrent_dropout_probabilities[
                    0
                ],
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
