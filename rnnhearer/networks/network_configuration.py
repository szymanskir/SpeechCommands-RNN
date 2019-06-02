from configparser import ConfigParser
from dataclasses import dataclass
from typing import List

from .audio_representation import AudioRepresentation


@dataclass
class NetworkConfiguration:
    name: str
    units_per_layer: List[int]
    representation: AudioRepresentation
    dropout_probabilities: List[float]
    recurrent_dropout_probabilities: List[float]
    extra_dense_layers: List[int]
    epochs_count: int
    batch_size: int

    @classmethod
    def from_config(cls, config_parser: ConfigParser):
        config = config_parser["DEFAULT"]
        name = config.get("Name")
        units_per_layer = [int(units) for units in config["UnitsPerLayer"].split(",")]
        representation = AudioRepresentation(config["SignalRepresentation"])
        dropout_probabilities = [
            float(prob) for prob in config["DropoutProbabilities"].split(",")
        ]
        recurrent_dropout_probabilities = [
            float(prob) for prob in config["RecurrentDropoutProbabilities"].split(",")
        ]
        extra_dense_layers = (
            [int(num_units) for num_units in config["ExtraDenseLayers"].split(",")]
            if config_parser.has_option("DEFAULT", "ExtraDenseLayers")
            else []
        )
        epochs_count = config.getint("EpochsCount")
        batch_size = config.getint("BatchSize")

        return cls(
            name,
            units_per_layer,
            representation,
            dropout_probabilities,
            recurrent_dropout_probabilities,
            extra_dense_layers,
            epochs_count,
            batch_size,
        )
