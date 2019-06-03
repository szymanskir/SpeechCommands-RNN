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
    def __init__(
        self,
        audio_representation: AudioRepresentation,
        kept_labels: List[str],
        sample_rate: int = 16000,
    ):
        self._converter = AudioRepresentationConverterFactory.create_converter(
            audio_representation
        )
        self._encoder = LabelEncoder()
        self._num_classes = len(kept_labels)
        self._encoder.fit(kept_labels)
        self._sample_rate = sample_rate

    def _read_wavfile(self, sample_filepath):
        file_data = wavfile.read(sample_filepath)
        audio = file_data[1]
        return trim_pad_audio(audio, self._sample_rate)

    def get_data_shape(self, sample_filepath: Path):
        converted_audio = self._converter.convert_audio_signal(
            [self._read_wavfile(sample_filepath)], self._sample_rate
        )[0]
        return converted_audio.shape

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        audio = amplitude_scaling(audio)
        audio = shift_audio(audio)
        audio = trim_pad_audio(audio, self._sample_rate)

        return audio

    def flow(
        self,
        samples: List[Tuple[Path, str]],
        batch_size: int,
        enable_preprocessing: bool,
    ):
        while True:
            for chunk in chunks(samples, batch_size):
                files = [self._read_wavfile(path) for path, _ in chunk]

                if enable_preprocessing:
                    files = [self._preprocess(audio) for audio in files]

                converted = self._converter.convert_audio_signal(
                    files, self._sample_rate
                )
                labels = [label for _, label in chunk]
                X = np.concatenate([converted])
                y = to_categorical(self._encoder.transform(labels), self._num_classes)

                yield X, y

    def flow_in_memory(
        self,
        samples: List[Tuple[Path, str]],
        batch_size: int,
        enable_preprocessing: bool,
    ):
        data = []
        for chunk in chunks(samples, batch_size):
            files = [self._read_wavfile(path) for path, _ in chunk]

            if enable_preprocessing:
                files = [self._preprocess(audio) for audio in files]

            converted = self._converter.convert_audio_signal(files, self._sample_rate)
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


def trim_pad_audio(audio, sample_rate):
    duration = len(audio)
    if duration < sample_rate:
        audio = np.pad(audio, (sample_rate - audio.size, 0), mode="constant")
    elif duration > sample_rate:
        audio = audio[0:sample_rate]

    return audio


def amplitude_scaling(audio, multiplier=0.2):
    return audio * np.random.uniform(1.0 - multiplier, 1.0 + multiplier)


def shift_audio(audio, ms_shift=100):
    ms = 16
    time_shift_dist = int(np.random.uniform(-(ms_shift * ms), (ms_shift * ms)))
    audio = np.roll(audio, time_shift_dist)

    return audio


def history_to_df(history: History) -> pd.DataFrame:
    history_values: dict = history.history
    history_values["model_name"] = history.model.name

    history_df = pd.DataFrame.from_dict(history_values)
    return history_df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
