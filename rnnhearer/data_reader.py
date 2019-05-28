import scipy.io.wavfile as wavfile
import logging
import multiprocessing
import numpy as np
from os import listdir
from os.path import isdir, join, basename
from pathlib import Path
from typing import Dict, List

_LOGGER = logging.getLogger(__name__)

_SPEECH_COMMANDS_SAMPLES_DIRECTORIES = {
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "four",
    "go",
    "happy",
    "house",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
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
    ) -> List[Dict[np.array, str]]:
        _LOGGER.info(f"Reading samples from {word_samples_dir}...")
        word_audio_samples = self._find_all_wav_files(
            join(self._audio_source, word_samples_dir)
        )
        label = basename(word_samples_dir)
        return [
            self._create_single_record(audio_data_file, label)
            for audio_data_file in word_audio_samples
        ]

    def read(self) -> List[Dict[np.array, str]]:
        word_samples_dir = list(
            filter(
                lambda x: x in _SPEECH_COMMANDS_SAMPLES_DIRECTORIES,
                listdir(self._audio_source),
            )
        )
        pool = multiprocessing.Pool()
        return sum(
            list(map(self._read_single_word_samples_dir, word_samples_dir)), []
        )

    @staticmethod
    def _find_all_wav_files(dir: str):
        return list(Path(dir).glob("*.wav"))
