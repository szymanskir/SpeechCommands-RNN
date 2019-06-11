from configparser import ConfigParser
from dataclasses import dataclass
from typing import List

from .audio_representation import AudioRepresentation


@dataclass
class NetworkConfiguration:
    name: str
    units_per_layer: List[int]
    representation: AudioRepresentation
    conv_layers_count: int
    dropout_probabilities: List[float]
    recurrent_dropout_probabilities: List[float]
    epochs_count: int
    batch_size: int

    @classmethod
    def from_config(cls, config_parser: ConfigParser):
        config = config_parser["DEFAULT"]
        name = config.get("Name")
        units_per_layer = [int(units) for units in config["UnitsPerLayer"].split(",")]
        representation = AudioRepresentation(config["SignalRepresentation"])
        conv_layers_count = (
            int(config["ConvLayersCount"])
            if config_parser.has_option("DEFAULT", "ConvLayersCount")
            else 0
        )
        dropout_probabilities = [
            float(prob) for prob in config["DropoutProbabilities"].split(",")
        ]
        recurrent_dropout_probabilities = [
            float(prob) for prob in config["RecurrentDropoutProbabilities"].split(",")
        ]
        epochs_count = config.getint("EpochsCount")
        batch_size = config.getint("BatchSize")

        return cls(
            name,
            units_per_layer,
            representation,
            conv_layers_count,
            dropout_probabilities,
            recurrent_dropout_probabilities,
            epochs_count,
            batch_size,
        )
