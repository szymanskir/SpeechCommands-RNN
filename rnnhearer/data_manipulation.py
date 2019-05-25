import numpy as np
import pandas as pd
from keras.callbacks import History
from keras.utils import to_categorical
from typing import List
from sklearn.preprocessing import LabelEncoder


def encode_categorical_labels(labels: List[str], kept_labels: List[str]) -> np.ndarray:
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
