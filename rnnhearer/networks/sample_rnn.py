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
    model.add(
        layers.CuDNNLSTM(
            units=network_configuration.units_per_layer[0],
            input_shape=input_shape,
            return_sequences=should_return_sequence[0],
        )
    )

    layers_count = len(network_configuration.units_per_layer)
    for layer_num in range(1, layers_count):
        layer = layers.CuDNNLSTM(
            units=network_configuration.units_per_layer[layer_num],
            return_sequences=should_return_sequence[layer_num],
        )
        model.add(layer)
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
