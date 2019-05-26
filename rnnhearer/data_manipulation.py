import numpy as np
from keras.utils import to_categorical
from typing import List
from sklearn.preprocessing import LabelEncoder


def encode_categorical_labels(labels: List[str], kept_labels: List[str]) -> np.ndarray:
    label_encoder = LabelEncoder()
    encoded_labels = to_categorical(
        y=label_encoder.fit_transform(labels), num_classes=len(kept_labels)
    )

    return encoded_labels
