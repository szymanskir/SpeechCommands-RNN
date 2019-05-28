import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd
from keras.callbacks import History
from keras.utils import to_categorical
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Set

from .networks import AudioRepresentation, AudioRepresentationConverterFactory


class AudioDataGenerator:
    @staticmethod
    def get_data_shape(
        sample_filepath: Path, audio_representation: AudioRepresentation
    ):
        converter = AudioRepresentationConverterFactory.create_converter(
            audio_representation
        )
        converted_sample = converter.convert_audio_signal(
            [wavfile.read(sample_filepath)]
        )[0]
        return converted_sample.shape

    @staticmethod
    def flow(
        samples: List[Tuple[Path, str]],
        audio_representation: AudioRepresentation,
        kept_labels: Set[str],
        batch_size: int,
    ):
        converter = AudioRepresentationConverterFactory.create_converter(
            audio_representation
        )

        for chunk in chunks(samples, batch_size):
            files = [wavfile.read(path) for path, _ in chunk]

            converted = converter.convert_audio_signal(files)
            labels = [label for _, label in chunk]
            yield np.concatenate([converted]), encode_categorical_labels(
                labels=labels, kept_labels=kept_labels
            )


def encode_categorical_labels(labels: List[str], kept_labels: Set[str]) -> np.ndarray:
    labels = [label if label in kept_labels else "unknown" for label in labels]
    label_encoder = LabelEncoder()
    encoded_labels = to_categorical(
        y=label_encoder.fit_transform(labels), num_classes=len(kept_labels)
    )

    return encoded_labels


def history_to_df(history: History) -> pd.DataFrame:
    history_values: dict = history.history
    history_values["model_name"] = history.model.name

    history_df = pd.DataFrame.from_dict(history_values)
    return history_df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
