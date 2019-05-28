import scipy.io.wavfile as wavfile
import logging
import numpy as np
from os import listdir
from os.path import isdir, join, basename
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from .data_manipulation import encode_categorical_labels

_LOGGER = logging.getLogger(__name__)

_SPEECH_COMMANDS_SAMPLES_DIRECTORIES = {
    "bed",
    "yes",
    "bird",
    "no",
    "cat",
    "up",
    "dog",
    "down",
    "down",
    "eight",
    "five",
    "four",
    "go",
    "happy",
    "house",
    "left",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "right",
    "seven",
    "on",
    "sheila",
    "off",
    "six",
    "stop",
    "stop",
    "three",
    "go",
    "tree",
    "two",
    "up",
    "wow",
    "yes",
    "zero",
}


class DataReader:
    def __init__(self, filepath: str):
        self._audio_source = f"{filepath}/audio"
        self._validate_input()

    def _validate_input(self):
        if not isdir(self._audio_source):
            raise FileNotFoundError("No audio file found in the dataset")
        self._check_if_contains_samples()

    def _check_if_contains_samples(self):
        sample_directories = set(listdir(self._audio_source))
        missing_directories = _SPEECH_COMMANDS_SAMPLES_DIRECTORIES - sample_directories
        if missing_directories:
            raise FileNotFoundError(
                f"Missing directories of samples: {missing_directories}"
            )

    def _create_single_record(
        self, audio_data_file: np.array, label: str
    ) -> Dict[np.array, str]:
        audio_data = wavfile.read(audio_data_file)
        return {"audio_data": audio_data, "label": label}

    def _read_single_word_samples_dir(
        self, word_samples_dir: str
    ) -> List[Tuple[Path, str]]:
        _LOGGER.info(f"Reading samples from {word_samples_dir}...")
        word_audio_samples = self._find_all_wav_files(
            join(self._audio_source, word_samples_dir)
        )
        label = basename(word_samples_dir)
        return [(audio_data_file, label) for audio_data_file in word_audio_samples]

    def read(self) -> List[Tuple[Path, str]]:
        word_samples_dir = list(
            filter(
                lambda x: x in _SPEECH_COMMANDS_SAMPLES_DIRECTORIES,
                listdir(self._audio_source),
            )
        )
        return sum(list(map(self._read_single_word_samples_dir, word_samples_dir)), [])

    def flow(self, input_x, converter) -> Iterable[Tuple[np.ndarray, str]]:
        main_labels = set(
            [
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
                "unknown",
            ]
        )

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i : i + n]

        for chunk in chunks(input_x, 32):
            files = [wavfile.read(path) for path, _ in chunk]
            labels = [
                label if label in main_labels else "unknown" for _, label in chunk
            ]

            converted = converter.convert_audio_signal(files)

            yield np.concatenate([converted]), encode_categorical_labels(
                labels=labels, kept_labels=main_labels
            )

    @staticmethod
    def _find_all_wav_files(dir: str):
        return list(Path(dir).glob("*.wav"))
