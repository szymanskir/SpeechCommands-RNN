import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd
from keras.callbacks import History
from keras.utils import to_categorical
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Set
import random
from .networks import AudioRepresentation, AudioRepresentationConverterFactory


class AudioDataGenerator:
    def __init__(
        self, audio_representation: AudioRepresentation, kept_labels: List[str]
    ):
        self._converter = AudioRepresentationConverterFactory.create_converter(
            audio_representation
        )
        self._encoder = LabelEncoder()
        self._num_classes = len(kept_labels)
        self._encoder.fit(kept_labels)

    def _read_wavfile(self, sample_filepath):
        file_data = wavfile.read(sample_filepath)
        samples = file_data[1]
        sr = file_data[0]
        if len(samples) >= sr:
            samples = samples
        else:
            samples = np.pad(
                samples,
                pad_width=(sr - len(samples), 0),
                mode="constant",
                constant_values=(0, 0),
            )

        return sr, samples

    def get_data_shape(self, sample_filepath: Path):

        converted_sample = self._converter.convert_audio_signal(
            [self._read_wavfile(sample_filepath)]
        )[0]
        return converted_sample.shape

    def flow(self, samples: List[Tuple[Path, str]], batch_size: int):
        random.shuffle(samples)
        while True:
            for chunk in chunks(samples, batch_size):
                files = [self._read_wavfile(path) for path, _ in chunk]

                converted = self._converter.convert_audio_signal(files)
                labels = [label for _, label in chunk]
                X = np.concatenate([converted])
                y = to_categorical(self._encoder.transform(labels), self._num_classes)

                yield X, y

    def flow_in_memory(self, samples: List[Tuple[Path, str]], batch_size: int):
        random.shuffle(samples)
        data = []
        for chunk in chunks(samples, batch_size):
            files = [self._read_wavfile(path) for path, _ in chunk]

            converted = self._converter.convert_audio_signal(files)
            labels = [label for _, label in chunk]
            data.append(
                (
                    np.concatenate([converted]),
                    to_categorical(
                        self._encoder.transform(labels), num_classes=self._num_classes
                    ),
                )
            )

        while True:
            for chunk in data:
                yield chunk


def history_to_df(history: History) -> pd.DataFrame:
    history_values: dict = history.history
    history_values["model_name"] = history.model.name

    history_df = pd.DataFrame.from_dict(history_values)
    return history_df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
