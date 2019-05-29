import numpy as np
from abc import ABCMeta, abstractmethod
from enum import auto
from python_speech_features import mfcc
from scipy.signal import resample, spectrogram
from strenum import StrEnum
from typing import List, Tuple, Type, Mapping


class AudioRepresentation(StrEnum):
    RAW = auto()
    SPECTROGRAM = auto()
    MFCC = auto()


class IAudioRepresentationConverter(metaclass=ABCMeta):
    @abstractmethod
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        raise NotImplementedError("subclasses must override foo()!")


class RawAudioRepresentationConverter(IAudioRepresentationConverter):
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        return [np.atleast_2d(audio_sample[1]) for audio_sample in audio_samples]


class SpectrogramAudioRepresentationConverter(IAudioRepresentationConverter):
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        return [spectrogram(audio_sample[1])[2] for audio_sample in audio_samples]


class MFCCAudioRepresentationConverter(IAudioRepresentationConverter):
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        return [
            mfcc(signal=audio_sample[1], samplerate=audio_sample[0])
            for audio_sample in audio_samples
        ]


class AudioRepresentationConverterFactory:
    @staticmethod
    def create_converter(
        audio_representaion: AudioRepresentation
    ) -> IAudioRepresentationConverter:
        converters: Mapping[auto, Type[IAudioRepresentationConverter]] = {
            AudioRepresentation.RAW: RawAudioRepresentationConverter,
            AudioRepresentation.SPECTROGRAM: SpectrogramAudioRepresentationConverter,
            AudioRepresentation.MFCC: MFCCAudioRepresentationConverter,
        }

        return converters[audio_representaion]()
