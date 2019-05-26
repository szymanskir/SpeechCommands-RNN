from enum import auto
from strenum import StrEnum


class AudioRepresentation(StrEnum):
    RAW = auto()
    SPECTOGRAM = auto()
    MFCC = auto()
