from configparser import ConfigParser
from dataclasses import dataclass
from typing import List

from .audio_representation import AudioRepresentation


@dataclass
class NetworkConfiguration:
    units_per_layer: List[int]
    representation: AudioRepresentation
    dropout_probabilities: List[float]
    epochs_count: int
    batch_size: int

    @classmethod
    def from_config(cls, config_parser: ConfigParser):
        config = config_parser["DEFAULT"]
        units_per_layer = [int(units) for units in config["UnitsPerLayer"].split(",")]
        representation = AudioRepresentation[config["SignalRepresentation"]]
        dropout_probabilities = [
            float(prob) for prob in config["DropoutProbabilities"].split(",")
        ]
        epochs_count = config.getint("EpochsCount")
        batch_size = config.getint("BatchSize")

        return cls(units_per_layer, representation, dropout_probabilities, epochs_count, batch_size)
