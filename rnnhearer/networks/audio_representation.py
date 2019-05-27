import numpy as np
from abc import ABCMeta, abstractmethod
from enum import auto
from python_speech_features import mfcc
from scipy.signal import spectrogram
from strenum import StrEnum
from typing import List, Tuple


class AudioRepresentation(StrEnum):
    SPECTROGRAM = auto()
    MFCC = auto()


class IAudioRepresentationConverter(metaclass=ABCMeta):
    @abstractmethod
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        pass


class SpectrogramAudioRepresentationConverter(IAudioRepresentationConverter):
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        return [
            spectrogram(x=audio_sample[1], fs=audio_sample[0])[2]
            for audio_sample in audio_samples
        ]


class MFCCAudioRepresentationConverter(IAudioRepresentationConverter):
    def convert_audio_signal(
        self, audio_samples: List[Tuple[np.ndarray, int]]
    ) -> List[np.ndarray]:
        return [
            mfcc(signal=audio_sample[1], samplerate=audio_samples[0])
            for audio_sample in audio_samples
        ]


class AudioRepresentationConverterFactory:
    @staticmethod
    def create_converter(
        audio_representaion: AudioRepresentation
    ) -> IAudioRepresentationConverter:
        converters = {
            AudioRepresentation.SPECTROGRAM: SpectrogramAudioRepresentationConverter,
            AudioRepresentation.MFCC: MFCCAudioRepresentationConverter,
        }

        constructor = converters[audio_representaion]
        return constructor()
