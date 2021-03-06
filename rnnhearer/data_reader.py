import logging
from os import listdir
from os.path import isdir, join, basename
from pathlib import Path
from typing import List, Tuple, Set
from collections import defaultdict
import random

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
    def __init__(self, data_dir: str, validation_file_path: str):
        self._audio_source = f"{data_dir}/audio"
        self._validation_file_path = validation_file_path
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

    def read(
        self, recognized_labels: Set[str], unknown_percentage: float = 0.1
    ) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
        word_samples_dir = list(
            filter(
                lambda x: x in _SPEECH_COMMANDS_SAMPLES_DIRECTORIES,
                listdir(self._audio_source),
            )
        )

        with open(self._validation_file_path) as f:
            validation_files = set(f.read().splitlines())

        train_samples: List[Tuple[Path, str]] = list()
        val_samples: List[Tuple[Path, str]] = list()
        for directory in word_samples_dir:
            _LOGGER.info(f"Reading samples from {directory}...")
            word_audio_samples = self._find_all_wav_files(
                join(self._audio_source, directory)
            )
            label = basename(directory)

            if label not in recognized_labels:
                label = "unknown"

            for audio_data_file in word_audio_samples:
                sample = (audio_data_file, label)
                if (
                    join(audio_data_file.parts[-2], audio_data_file.parts[-1])
                    not in validation_files
                ):
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

        samples_by_label = defaultdict(list)
        for path, label in train_samples:
            samples_by_label[label].append((path, label))

        not_unknown_count = sum(
            [
                len(samples)
                for label, samples in samples_by_label.items()
                if label != "unknown"
            ]
        )
        unknown_count = len(samples_by_label["unknown"])

        samples_by_label["unknown"] = random.sample(
            samples_by_label["unknown"], int(unknown_percentage * not_unknown_count)
        )

        rebalanced_train_samples = [
            item for sublist in samples_by_label.values() for item in sublist
        ]

        return rebalanced_train_samples, val_samples

    @staticmethod
    def _find_all_wav_files(dir: str):
        return list(Path(dir).glob("*.wav"))
